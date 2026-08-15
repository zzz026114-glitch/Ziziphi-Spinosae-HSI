import numpy as np
import joblib


TARGET_H = 208
TARGET_W = 292


class PCAProcessor:

    def __init__(
        self,
        scaler_path,
        pca_path
    ):

        print("Loading PCA model...")

        self.scaler = joblib.load(
            scaler_path
        )

        self.pca = joblib.load(
            pca_path
        )

        print("PCA loaded.")

    # =====================================================
    # Pad ROI
    # =====================================================
    def pad_roi(
        self,
        roi
    ):

        h, w, b = roi.shape

        if h > TARGET_H or w > TARGET_W:

            raise ValueError(
                f"ROI size exceeds target:"
                f"{roi.shape}"
            )

        out = np.zeros(
            (
                TARGET_H,
                TARGET_W,
                b
            ),
            dtype=np.float32
        )

        sy = (
            TARGET_H - h
        ) // 2

        sx = (
            TARGET_W - w
        ) // 2

        out[
            sy:sy+h,
            sx:sx+w
        ] = roi

        return out

    # =====================================================
    # PCA Transform
    # =====================================================
    def transform(
        self,
        roi
    ):

        roi = self.pad_roi(
            roi
        )

        H, W, B = roi.shape

        pixels = roi.reshape(
            -1,
            B
        )

        pixels_norm = (
            self.scaler.transform(
                pixels
            )
        )

        pixels_pca = (
            self.pca.transform(
                pixels_norm
            )
        )

        cube_pca = pixels_pca.reshape(
            H,
            W,
            -1
        )

        return (
            roi,
            cube_pca.astype(
                np.float32
            )
        )