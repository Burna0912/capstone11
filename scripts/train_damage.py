import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from efficientnet_pytorch import EfficientNet
from utils.transforms import get_transforms

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = datasets.ImageFolder("data/damage", transform=get_transforms())
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(model._fc.in_features, 2)  # intact, damaged
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[damage] Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "models/damage_efficientnet.pt")

if __name__ == "__main__":
    train()
