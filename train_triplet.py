import torch
import torch.nn as nn
import torch.nn.functional as F
from model_loader import load_embedding_model
from triplet_dataset import TripletDataset
from config import cfg  # o carga manual desde config.yaml

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_embedding_model(cfg["embedding_model"]).to(device)

dataset = TripletDataset(
    cadena_dir=cfg["cadena_dir"],
    normal_dir=cfg["normal_dir"],
    image_size=cfg["image_size"],
    mean=cfg["normalize_mean"],
    std=cfg["normalize_std"]
)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.TripletMarginLoss(margin=1.0)

for epoch in range(10):
    model.train()
    total_loss = 0
    for anchor, positive, negative in loader:
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)
        loss = loss_fn(emb_a, emb_p, emb_n)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")