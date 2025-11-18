import os
from PIL import Image
import random
import torchvision.transforms as T
import torch

class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, cadena_dir, normal_dir, image_size, mean, std):
        self.cadena = [os.path.join(cadena_dir, f) for f in os.listdir(cadena_dir)]
        self.normal = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir)]
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.cadena)

    def __getitem__(self, idx):
        anchor = Image.open(self.cadena[idx]).convert("RGB")
        positive = Image.open(random.choice(self.cadena)).convert("RGB")
        negative = Image.open(random.choice(self.normal)).convert("RGB")
        return self.transform(anchor), self.transform(positive), self.transform(negative)