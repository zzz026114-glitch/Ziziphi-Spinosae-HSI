import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segmentation_1 import segment_single_hsi
import pandas as pd
from pca import PCAProcessor

from predictor import Predictor

class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        # ==========================
        # 删除历史缓存图
        # ==========================
        for f in ["temp_roi.png", "temp_spectrum.png"]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

        self.setWindowTitle(
            "Workflow: High-Throughput Identification System"
        )

        self.resize(1600, 900)

        self.init_ui()
        self.init_ui()

        self.btn_open.clicked.connect(
            self.select_hsi
        )

        self.btn_model.clicked.connect(
            self.select_model
        )

        self.btn_run.clicked.connect(
            self.run_analysis
        )

        self.btn_export_csv.clicked.connect(
            self.export_csv
        )

        self.btn_export_fig.clicked.connect(
            self.export_figure
        )
        print("Loading PCA...")

        self.pca_processor = PCAProcessor(

            scaler_path=
            r"E:\data\suanzaoren\stage1\PCA\pca_models_new\scaler_train.pkl",

            pca_path=
            r"E:\data\suanzaoren\stage1\PCA\pca_models_new\pca_train.pkl"
        )

        print("Loading Model...")

        self.predictor =None

        self.roi_spectra = []

        self.roi_boxes = []

        self.current_roi_index = -1

        self.current_cube = None
#clear
    def clear_visualization(self):

        # ROI
        self.roi_view.clear()

        self.roi_view.setPixmap(QPixmap())

        self.roi_view.setText("No ROI loaded")

        # Spectrum
        self.spectrum_view.clear()

        self.spectrum_view.setPixmap(QPixmap())

        self.spectrum_view.setText(
            "No Spectrum available"
        )

        # Gallery
        self.gallery.clear()

        # 删除缓存图
        for f in [
            "temp_roi.png",
            "temp_spectrum.png"
        ]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass
    # =================================================
    # 选择文件
    # =================================================
    def select_hsi(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select HSI File",
            "",
            "MAT Files (*.mat)"
        )

        if file_path:
            # 先清空旧结果
            self.clear_visualization()

            self.current_cube = None

            self.current_roi_index = -1

            self.roi_spectra = []

            self.roi_boxes = []

            self.hsi_path = file_path

            self.file_label.setText(
                os.path.basename(file_path)
            )

    # =================================================
    # model
    # =================================================
    def select_model(self):

        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model",
            "",
            "PyTorch Model (*.pth)"
        )

        if model_path:
            self.model_path = model_path

            self.model_label.setText(
                os.path.basename(model_path)
            )

            self.predictor = Predictor(
                model_path=model_path
            )

    # =================================================
    #  roi缩略图
    # =================================================
    def create_roi_thumbnail(self, roi):

        # roi: (H, W, C)

        # 取中间波段
        band = roi[:, :, roi.shape[2] // 2]

        band = cv2.normalize(
            band,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        h, w = band.shape

        qimg = QImage(
            band.data,
            w,
            h,
            w,
            QImage.Format_Grayscale8
        )

        pix = QPixmap.fromImage(qimg)

        # 固定缩略图大小（论文风格关键）
        pix = pix.scaled(
            120,
            120,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        return pix.toImage()
    # =================================================
    #  Gallery
    # =================================================
    def on_roi_selected(
            self,
            index
    ):

        if index < 0:
            return

        self.current_roi_index = index

        spectrum = self.roi_spectra[index]

        self.show_spectrum(
            spectrum
        )

        self.show_roi_overview()

    # =================================================
    #  Run analysis
    # =================================================

    def run_analysis(self):
        if not hasattr(self, "hsi_path"):
            QMessageBox.warning(
                self,
                "Warning",
                "Please select HSI file first."
            )
            return

        # -------------------------------
        # 清空 ROI 和 Spectrum 显示
        # -------------------------------
        self.roi_view.clear()
        self.roi_view.setText("No ROI loaded")
        self.roi_view.setStyleSheet("""
            background:white;
            border:1px solid #D9D9D9;
            border-radius:12px;
            color:#AAAAAA;
            font-size:16px;
        """)

        self.spectrum_view.clear()
        self.spectrum_view.setText("No Spectrum available")
        self.spectrum_view.setStyleSheet("""
            background:white;
            border:1px solid #D9D9D9;
            border-radius:12px;
            color:#AAAAAA;
            font-size:16px;
        """)

        try:

            self.clear_visualization()

            QApplication.setOverrideCursor(
                Qt.WaitCursor
            )

            # =====================================
            # segmentation
            # =====================================
            cube, mask, roi_list, boxes = segment_single_hsi(self.hsi_path)
            self.current_cube = cube
            self.roi_boxes = boxes
            self.roi_spectra.clear()
            self.gallery.clear()
            self.roi_num_label.setText(f"ROI: {len(roi_list)}")

            true_count = 0
            false_count = 0
            confidences = []
            self.export_results = []

            # =====================================
            # predict
            # =====================================
            for i, roi in enumerate(roi_list):
                mask_roi = np.sum(roi, axis=2) > 0
                pixels = roi[mask_roi]
                mean_spec = pixels.mean(axis=0)
                self.roi_spectra.append(mean_spec)

                roi127, roi25 = self.pca_processor.transform(roi)
                result = self.predictor.predict(roi127, roi25)
                label = result["label"]
                conf = result["confidence"]
                confidences.append(conf)

                if label == "TRUE":
                    true_count += 1
                else:
                    false_count += 1

                self.export_results.append([i + 1, label, conf])

                # =========================
                # ROI thumbnail
                # =========================
                thumb = self.create_roi_thumbnail(roi)
                pixmap = QPixmap.fromImage(thumb)
                item = QListWidgetItem()
                item.setIcon(QIcon(pixmap))
                item.setText(f"ROI {i + 1}\n{label}\n{conf * 100:.1f}%")
                item.setTextAlignment(Qt.AlignHCenter)
                item.setSizeHint(QSize(130, 160))
                self.gallery.addItem(item)

            # =====================================
            # summary
            # =====================================
            self.true_label.setText(f"TRUE: {true_count}")
            self.false_label.setText(f"FALSE: {false_count}")
            avg_conf = np.mean(confidences) if confidences else 0
            self.conf_label.setText(f"{avg_conf * 100:.2f}%")

            # 自动选中第一个 ROI
            if self.gallery.count() > 0:
                self.gallery.setCurrentRow(0)
                self.on_roi_selected(0)

            QMessageBox.information(
                self,
                "Finished",
                "Analysis Completed."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                str(e)
            )

        finally:
            QApplication.restoreOverrideCursor()
    #csv
    def export_csv(self):

        if not hasattr(self, "export_results"):
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            "",
            "CSV (*.csv)"
        )

        if not path:
            return

        df = pd.DataFrame(
            self.export_results,
            columns=[
                "ROI_ID",
                "Prediction",
                "Confidence"
            ]
        )

        df.to_csv(
            path,
            index=False
        )
    #figure
    def export_figure(self):

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            "",
            "PNG (*.png)"
        )

        if not path:
            return

        if os.path.exists("temp_roi.png"):
            import shutil

            shutil.copy(
                "temp_roi.png",
                path
            )
    #step
    def create_step_title(self, text):

        label = QLabel(text)

        label.setStyleSheet("""
            color:#1565C0;
            font-size:13px;
            font-weight:700;
            padding-top:8px;
            padding-bottom:4px;
        """)

        return label
    #roi area
    def resizeEvent(self, event):

        super().resizeEvent(event)

        # 有ROI图时才刷新
        if self.roi_view.pixmap() is not None:
            pix = self.roi_view.pixmap()

            self.roi_view.setPixmap(
                pix.scaled(
                    int(self.roi_view.width() * 0.9),
                    int(self.roi_view.height() * 0.9),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

        # 有光谱图时才刷新
        if self.spectrum_view.pixmap() is not None:
            pix = self.spectrum_view.pixmap()

            self.spectrum_view.setPixmap(
                pix.scaled(
                    self.spectrum_view.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )


    def create_card(self):

        card = QFrame()

        card.setObjectName("card")

        card.setStyleSheet("""
        QFrame#card{

            background:white;

            border-radius:16px;

            border:1px solid #E6EAF0;
        }
        """)

        shadow = QGraphicsDropShadowEffect()

        shadow.setBlurRadius(30)

        shadow.setOffset(0, 4)

        shadow.setColor(
            QColor(0, 0, 0, 30)
        )

        card.setGraphicsEffect(
            shadow
        )

        return card

    def closeEvent(self, event):

        for f in [
            "temp_roi.png",
            "temp_spectrum.png"
        ]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except:
                    pass

        event.accept()
    # =================================================
    # UI
    # =================================================
    def init_ui(self):
        central = QWidget()
        central.setStyleSheet("background:#F4F6F8;")
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(15)
        root_layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header = QLabel(
            "High-Throughput Identification System of "
            "Ziziphi Spinosae Semen Based on HSI"
        )
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("""
            QLabel{
                background:white;
                border-radius:16px;
                font-size:24px;
                font-weight:700;
                padding:15px;
            }
        """)
        root_layout.addWidget(header)

        # Main Layout
        main_layout = QHBoxLayout()
        main_layout.setSpacing(15)
        root_layout.addLayout(main_layout)

        # LEFT PANEL
        self.left_panel = self.build_left_panel()
        self.left_panel.setMaximumWidth(320)
        main_layout.addWidget(self.left_panel, 1)

        # RIGHT PANEL
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, 4)

        # TOP AREA: ROI + Spectrum
        top_layout = QHBoxLayout()
        right_layout.addLayout(top_layout, 3)

        # ROI CARD
        roi_card = self.create_card()
        roi_layout = QVBoxLayout(roi_card)
        roi_title = QLabel("ROI Detection Result")
        roi_title.setAlignment(Qt.AlignCenter)
        roi_title.setStyleSheet("font-size:18px;font-weight:600;")
        roi_layout.addWidget(roi_title)

        self.roi_view = QLabel()
        self.roi_view.setAlignment(Qt.AlignCenter)
        self.roi_view.setMinimumSize(600, 420)
        self.roi_view.setText("No ROI loaded")
        self.roi_view.setStyleSheet("""
            background:white;
            border:1px solid #D9D9D9;
            border-radius:12px;
            color:#AAAAAA;
            font-size:16px;
        """)
        roi_layout.addWidget(self.roi_view, 1)
        top_layout.addWidget(roi_card, 3)

        # SPECTRUM CARD
        spectrum_card = self.create_card()
        spec_layout = QVBoxLayout(spectrum_card)
        spec_title = QLabel("Mean Spectrum")
        spec_title.setAlignment(Qt.AlignCenter)
        spec_title.setStyleSheet("font-size:18px;font-weight:600;")
        spec_layout.addWidget(spec_title)

        self.spectrum_view = QLabel()
        self.spectrum_view.setAlignment(Qt.AlignCenter)
        self.spectrum_view.setMinimumSize(400, 300)
        self.spectrum_view.setText("No Spectrum available")
        self.spectrum_view.setStyleSheet("""
            background:white;
            border:1px solid #D9D9D9;
            border-radius:12px;
            color:#AAAAAA;
            font-size:16px;
        """)
        spec_layout.addWidget(self.spectrum_view, 1)
        top_layout.addWidget(spectrum_card, 2)

        # BOTTOM AREA: Gallery
        gallery_card = self.create_card()
        gallery_layout = QVBoxLayout(gallery_card)
        gallery_title = QLabel("ROI Gallery")
        gallery_title.setAlignment(Qt.AlignCenter)
        gallery_title.setStyleSheet("font-size:18px;font-weight:600;padding:5px;")
        gallery_layout.addWidget(gallery_title)

        self.gallery = QListWidget()
        self.gallery.setViewMode(QListView.IconMode)
        self.gallery.setResizeMode(QListView.Adjust)
        self.gallery.setMovement(QListView.Static)
        self.gallery.setFlow(QListView.LeftToRight)
        self.gallery.setWrapping(True)
        self.gallery.setSpacing(15)
        self.gallery.setGridSize(QSize(140, 180))
        self.gallery.setIconSize(QSize(100, 100))
        self.gallery.setStyleSheet("""
            QListWidget{background:white;border:none;}
            QListWidget::item{border:1px solid #E6EAF0;border-radius:10px;padding:5px;}
            QListWidget::item:selected{border:2px solid #1565C0;}
        """)
        gallery_layout.addWidget(self.gallery)
        right_layout.addWidget(gallery_card, 4)

        # SIGNAL
        self.gallery.currentRowChanged.connect(self.on_roi_selected)

    # ===========================
    # ROI 显示函数
    # ===========================
    def show_roi_overview(self):
        if self.current_cube is None or not self.roi_boxes:
            return

        band = self.current_cube[:, :, self.current_cube.shape[2] // 2]

        plt.figure(figsize=(8, 5))
        plt.imshow(band, cmap="gray")
        ax = plt.gca()

        for i, box in enumerate(self.roi_boxes):
            x, y, w, h = box
            color = "yellow" if i == self.current_roi_index else "red"
            lw = 4 if i == self.current_roi_index else 2

            rect = plt.Rectangle(
                (x, y),
                w,
                h,
                edgecolor=color,
                facecolor="none",
                linewidth=lw
            )
            ax.add_patch(rect)

        plt.axis("off")
        temp_png = "temp_roi.png"
        plt.savefig(temp_png, dpi=200, bbox_inches="tight", pad_inches=0.02, facecolor="white")
        plt.close()

        pix = QPixmap(temp_png)
        # 缩放到 ROI QLabel 90%
        target_size = QSize(int(self.roi_view.width() * 0.9), int(self.roi_view.height() * 0.9))
        self.roi_view.setPixmap(
            pix.scaled(
                target_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
        self.roi_view.setText("")

    # ===========================
    # Spectrum 显示函数
    # ===========================
    def show_spectrum(self, spectrum):
        if spectrum is None:
            return

        plt.figure(figsize=(5, 3))
        wavelength = np.linspace(381, 1036, len(spectrum))  # 横轴波长
        plt.plot(wavelength, spectrum, linewidth=2.5)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectance")
        plt.grid(alpha=0.3)
        plt.box(False)
        plt.tight_layout()

        temp_png = "temp_spectrum.png"
        plt.savefig(temp_png, dpi=150)
        plt.close()

        pix = QPixmap(temp_png)
        self.spectrum_view.setPixmap(
            pix.scaled(
                self.spectrum_view.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
        self.spectrum_view.setText("")

    def build_left_panel(self):

        frame = self.create_card()


        layout = QVBoxLayout(
            frame
        )

        self.conf_label = QLabel(
            "Confidence: 0%"
        )

        layout.addWidget(
            self.conf_label
        )
        # ----------------------------
        workflow_title = QLabel("Workflow")

        workflow_title.setStyleSheet("""
                font-size:20px;
                font-weight:700;
                padding:8px;
                """)

        layout.addWidget(workflow_title)

        self.btn_open = QPushButton(
            "Select HSI"
        )

        self.btn_model = QPushButton(
            "Select Model"
        )

        self.model_label = QLabel(
            "No Model"
        )

        self.btn_run = QPushButton(
            "Run Analysis"
        )

        self.btn_export_csv = QPushButton(
            "Export CSV"
        )

        self.btn_export_fig = QPushButton(
            "Export Figure"
        )
        # ----------------------------

        self.file_label = QLabel(
            "No file"
        )

        # ----------------------------

        self.roi_num_label = QLabel(
            "ROI: 0"
        )

        self.true_label = QLabel(
            "TRUE: 0"
        )


        self.false_label = QLabel(
            "FALSE: 0"
        )

        # ----------------------------

        self.result_label = QLabel(
            ""
        )

        self.result_label.setAlignment(
            Qt.AlignCenter
        )

        self.result_label.setStyleSheet(
            """
            font-size:28px;
            font-weight:bold;
            color:#1565C0;
            """
        )

        # step1
        layout.addWidget(
            self.create_step_title("STEP 1")
        )

        layout.addWidget(
            QLabel("Import HSI")
        )

        layout.addWidget(self.btn_open)

        layout.addWidget(self.file_label)
        # step2
        layout.addWidget(
            self.create_step_title("STEP 2")
        )

        layout.addWidget(
            QLabel("Load CNN Model")
        )

        layout.addWidget(self.btn_model)

        layout.addWidget(self.model_label)
        # step3
        layout.addWidget(
            self.create_step_title("STEP 3")
        )

        layout.addWidget(
            QLabel("Run Identification")
        )

        layout.addWidget(self.btn_run)
        # line
        line = QFrame()

        line.setFrameShape(
            QFrame.HLine
        )

        line.setStyleSheet(
            "color:#DDDDDD;"
        )

        layout.addWidget(line)
        # step4
        step4 = QLabel("STEP 4")

        step4.setStyleSheet("""
                font-size:13px;
                font-weight:700;
                color:#1565C0;
                """)

        layout.addWidget(step4)

        layout.addWidget(QLabel("Export Results"))

        layout.addWidget(self.btn_export_csv)

        layout.addWidget(self.btn_export_fig)
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)

        layout.addWidget(line2)

        stats_title = QLabel("Statistics")
        stats_title.setStyleSheet("""
        font-size:16px;
        font-weight:700;
        padding-top:8px;
        """)

        layout.addWidget(stats_title)

        layout.addWidget(self.roi_num_label)
        layout.addWidget(self.true_label)
        layout.addWidget(self.false_label)
        layout.addWidget(self.conf_label)



        layout.addStretch()

        layout.addWidget(line2)

        layout.addWidget(
            QLabel("Final Result")
        )

        layout.addWidget(
            self.result_label
        )
        button_style = """
        QPushButton{

            font-size:15px;

            font-weight:600;

            min-height:42px;

            border:1px solid #D0D0D0;

            border-radius:6px;

            background:white;
        }

        QPushButton:hover{

            background:#F5F5F5;
        }
        """
        for btn in [

            self.btn_open,
            self.btn_model,
            self.btn_run,
            self.btn_export_csv,
            self.btn_export_fig

        ]:
            btn.setStyleSheet(button_style)
        return frame


if __name__ == "__main__":

    app = QApplication(sys.argv)

    font = QFont(
        "Arial",
        10
    )

    app.setFont(font)
    win = MainWindow()

    win.show()

    sys.exit(
        app.exec_()
    )