import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_yolo import ObjectDetectionDataset
from yolo_loss import YOLOLoss

class YOLOModel(nn.Module):
    def __init__(self, num_classes=20, grid_size=7, bbox_per_cell=2):
        super(YOLOModel, self).__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.bbox_per_cell = bbox_per_cell
        self.output_size = bbox_per_cell * 5 + num_classes
        self.conv = nn.Conv2d(3, self.output_size, kernel_size=1)

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        out = self.conv(x)
        return out.view(batch_size, self.grid_size, self.grid_size, self.output_size)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
GRID_SIZE = 7
NUM_CLASSES = 3
ROOT_DIR = "\yolo"

transform = None
dataset = ObjectDetectionDataset(root_dir=ROOT_DIR, transform=transform)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=lambda x: x
)

model = YOLOModel(num_classes=NUM_CLASSES, grid_size=GRID_SIZE).to(DEVICE)
criterion = YOLOLoss().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train(model, dataloader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    train(model, dataloader, criterion, optimizer, DEVICE, NUM_EPOCHS)
