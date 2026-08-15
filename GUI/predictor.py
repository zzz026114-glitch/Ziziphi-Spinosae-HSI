import torch
import numpy as np

from CrossSSA_new import SpectralSpatialNet


class Predictor:

    def __init__(
        self,
        model_path,
        device=None
    ):

        if device is None:

            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = device

        print(
            f"Loading model on {self.device}"
        )

        self.model = SpectralSpatialNet(
            num_classes=2
        ).to(
            self.device
        )

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.eval()

        print("Model loaded.")

    # =====================================================
    # Predict
    # =====================================================
    def predict(
        self,
        roi127,
        roi25
    ):

        # (H,W,C)
        # ->
        # (C,H,W)

        roi127 = np.transpose(
            roi127,
            (2,0,1)
        )

        roi25 = np.transpose(
            roi25,
            (2,0,1)
        )

        raw_tensor = torch.tensor(
            roi127,
            dtype=torch.float32
        ).unsqueeze(0)

        pca_tensor = torch.tensor(
            roi25,
            dtype=torch.float32
        ).unsqueeze(0)

        raw_tensor = raw_tensor.to(
            self.device
        )

        pca_tensor = pca_tensor.to(
            self.device
        )

        with torch.no_grad():

            feat, logits = self.model(
                raw_tensor,
                pca_tensor
            )

            probs = torch.softmax(
                logits,
                dim=1
            )
            print(probs.cpu().numpy())

            pred = torch.argmax(
                probs,
                dim=1
            ).item()

            confidence = (
                probs[0, pred]
                .cpu()
                .numpy()
                .item()
            )

        label = (
            "TRUE"
            if pred == 1
            else "FALSE"
        )

        return {

            "label": label,

            "class_id": pred,

            "confidence": confidence
        }