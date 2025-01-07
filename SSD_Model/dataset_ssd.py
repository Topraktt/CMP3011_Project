import os
import torch
import cv2
from torch.utils.data import Dataset

class FaceMaskDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(320, 320)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Images folder missing: {self.images_dir}")
        
        self.image_files = [file for file in os.listdir(self.images_dir) if file.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.png', '.txt'))
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.target_size)
        img = torch.tensor(img / 255.0, dtype=torch.float32).permute(2, 0, 1)
        
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    labels.append(int(data[0]))
                    x, y, w, h = map(float, data[1:])
                    x_min = (x - w / 2) * self.target_size[0]
                    y_min = (y - h / 2) * self.target_size[1]
                    x_max = (x + w / 2) * self.target_size[0]
                    y_max = (y + h / 2) * self.target_size[1]
                    boxes.append([x_min, y_min, x_max, y_max])
        
        if not boxes:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
        
        return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
