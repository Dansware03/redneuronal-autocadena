import torch
import torchvision.transforms as T
import numpy as np
import cv2

class EmbeddingExtractor:
    def __init__(self, model, image_size: int, mean, std):
        self.model = model
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def extract_from_path(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"No se pudo leer la imagen: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            vec = self.model(tensor).squeeze().cpu().numpy()
        # Normalizamos para similitud por producto interior (cosine)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-12)