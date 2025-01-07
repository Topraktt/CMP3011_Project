import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

class ObjectDetectionDataset(Dataset):
    def __init__(self, dataset_path, image_transform=None):
        self.images_path = os.path.join(dataset_path, 'images')
        self.labels_path = os.path.join(dataset_path, 'labels')
        
        self.image_list = sorted(os.listdir(self.images_path))
        self.label_list = sorted(os.listdir(self.labels_path))
        self.transform = image_transform

    def __len__(self):
        return len(self.image_list)

    def _load_image(self, index):
        image_file = os.path.join(self.images_path, self.image_list[index])
        return Image.open(image_file).convert('RGB')

    def _load_label(self, index):
        label_file = os.path.join(self.labels_path, self.label_list[index])
        with open(label_file, 'r') as file:
            annotations = [list(map(float, line.strip().split())) for line in file]
        return torch.tensor(annotations, dtype=torch.float32)

    def __getitem__(self, index):
        image = self._load_image(index)
        labels = self._load_label(index)
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels

if __name__ == '__main__':
    from torchvision.transforms import Compose, Resize, ToTensor

    preprocessing = Compose([
        Resize((640, 640)),
        ToTensor()
    ])

    data_path = '\yolo'
    dataset = ObjectDetectionDataset(dataset_path=data_path, image_transform=preprocessing)
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda batch: batch)
    
    for images, labels in loader:
        print(f'Batch size: {len(images)}, Labels: {labels}')
