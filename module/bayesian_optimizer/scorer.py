import os
import requests
import torch
import torch.nn as nn
from typing import List
from pathlib import Path
from PIL import Image

LAION_URL = "https://github.com/Xerxemi/sdweb-auto-MBW/blob/master/scripts/classifiers/laion/"

CHAD_URL = "https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/"


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticScorer:
    """
    CLIP Aesthetic Score (LAION / CHAD) を用いて生成画像を評価するクラス
    """

    def __init__(self, method: str = "laion", device: str = "cuda"):
        self.method = method
        self.device = device

        # モデルの保存先 (SD-merger の models フォルダ内)
        self.model_dir = Path("models", "aesthetic")
        os.makedirs(self.model_dir, exist_ok=True)

        if self.method == "laion":
            self.scorer_model_name = "laion-sac-logos-ava-v2.safetensors"
        elif self.method == "chad":
            self.scorer_model_name = "ava+logos-l14-linearMSE.pth"
        else:
            raise ValueError(f"Unknown scorer method: {self.method}")

        self.model_path = self.model_dir / self.scorer_model_name
        self.model = None
        self.clip_model = None
        self.clip_preprocess = None

    def initialize(self):
        """モデルをダウンロードし、メモリにロードする"""
        self._get_model()
        self._load_model()

    def _get_model(self) -> None:
        if self.model_path.is_file():
            return

        import logging

        logging.info(f"Downloading aesthetic model from {self.method}...")

        if self.method == "chad":
            url = CHAD_URL
        elif self.method == "laion":
            url = LAION_URL

        url += f"{self.scorer_model_name}?raw=true"

        r = requests.get(url)
        r.raise_for_status()

        with open(self.model_path.absolute(), "wb") as f:
            f.write(r.content)
        logging.info(f"Downloaded aesthetic model to {self.model_path}")

    def _load_model(self) -> None:
        import logging

        logging.info(f"Loading {self.scorer_model_name}...")

        # 遅延インポート (必要時のみ依存関係をロード)
        import safetensors.torch
        import clip

        self.model = AestheticPredictor(768).to(self.device).eval()

        if self.model_path.suffix == ".safetensors":
            self.model.load_state_dict(
                safetensors.torch.load_file(
                    self.model_path,
                )
            )
            self.model.to(self.device)
        else:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.eval()

        # Load CLIP
        self.clip_model_name = "ViT-L/14"
        logging.info(f"Loading CLIP {self.clip_model_name}...")
        self.clip_model, self.clip_preprocess = clip.load(
            self.clip_model_name,
            device=self.device,
        )

    def _get_image_features(self, image: Image.Image) -> torch.Tensor:
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().detach().numpy()

    def score(self, image: Image.Image) -> float:
        """単一の画像をスコアリングする"""
        if self.model is None or self.clip_model is None:
            self.initialize()

        image_features = self._get_image_features(image)
        score_tensor = self.model(
            torch.from_numpy(image_features).to(self.device).float(),
        )

        return score_tensor.item()

    def batch_score(self, images: List[Image.Image]) -> float:
        """複数画像のスコア平均を返す"""
        if not images:
            return 0.0

        scores = [self.score(img) for (img) in images]
        return sum(scores) / len(scores)

    def unload(self):
        """VRAMを解放する"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.clip_model is not None:
            del self.clip_model
            self.clip_model = None
        if self.clip_preprocess is not None:
            del self.clip_preprocess
            self.clip_preprocess = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
