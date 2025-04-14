import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import mvit_v1_b
from torchvision.transforms import Resize, Normalize, Compose
from torch.amp import autocast, GradScaler
import av


def get_video_paths_and_labels(root_dir):
    
    video_paths = []
    labels = []
    class_to_idx = {}

    for i, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_dir = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_dir):
            continue

        class_to_idx[class_name] = i

        for file in os.listdir(class_dir):
            if file.endswith(('.mp4', '.mov')):
                video_paths.append(os.path.join(class_dir, file))
                labels.append(i)

    return video_paths, labels, class_to_idx

class MyVideoDataset(Dataset):
    def __init__(self, video_paths, labels, clip_len=16):
        self.video_paths = video_paths
        self.labels = labels
        self.clip_len = clip_len
        self.transform = Compose([
            Resize((224, 224)),
            Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video = self.load_video_tensor(self.video_paths[idx])
        label = self.labels[idx]
        video = video.permute(1, 0, 2, 3)
        return video, label

    def load_video_tensor(self, path):
        container = av.open(path)
        frames = []

        for frame in container.decode(video=0):
            img = frame.to_rgb().to_ndarray()
            img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
            img_tensor = self.transform(img_tensor)
            frames.append(img_tensor)

        video = torch.stack(frames)[:self.clip_len]
        if video.size(0) < self.clip_len:
            padding = torch.zeros(self.clip_len - video.size(0), 3, 224, 224)
            video = torch.cat((video, padding), dim=0)

        return video
    
def train():
    root_dir = "C:/Users/Dylan/Documents/School/Capstone/Training-Data"
    batch_size = 16
    num_epochs = 25
    lr = 1e-4
    clip_len = 16

    video_paths, labels, class_to_idx = get_video_paths_and_labels(root_dir)
    num_classes = len(class_to_idx)

    dataset = MyVideoDataset(video_paths, labels, clip_len=clip_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on: ", device)

    model = mvit_v1_b(weights=None)

    num_features = model.head[1].in_features
    model.head[1] = nn.Linear(num_features, num_classes)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for videos, labels in dataloader:
            videos = videos.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"mvit_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()