from pathlib import Path

import torch
import torch.nn.functional as F
import timm
import pandas as pd

from torchvision import transforms
from PIL import Image

from config.settings import CONFIDENCE_THRESHOLD, MARGIN_THRESHOLD


class AnimalPredictor:

    def __init__(self, project_root):

        self.project_root = Path(project_root)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        stats_path = self.project_root / "final_dataset_statistics.csv"
        df = pd.read_csv(stats_path)
        self.class_names = sorted(df["Species"].tolist())

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.model = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=len(self.class_names)
        )

        self.model.load_state_dict(
            torch.load(
                self.project_root / "models" / "best_model.pth",
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path):

        image = Image.open(image_path).convert("RGB")

        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=1)

        top5_prob, top5_idx = torch.topk(probabilities, 5)

        results = []

        for prob, idx in zip(top5_prob[0], top5_idx[0]):

            results.append({
                "class": self.class_names[idx],
                "confidence": round(prob.item() * 100, 2)
            })

        top1 = results[0]["confidence"]
        top2 = results[1]["confidence"]

        if top1 < (CONFIDENCE_THRESHOLD * 100) or (top1 - top2) < (MARGIN_THRESHOLD * 100):

            return {
                "prediction": "Unclassified",
                "confidence": top1,
                "top5": results
            }

        return {
            "prediction": results[0]["class"],
            "confidence": top1,
            "top5": results
        }