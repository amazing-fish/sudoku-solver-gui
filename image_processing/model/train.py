import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

from image_processing.model.sudoku_resnet import SudokuResNet
from dataset import SudokuDataset  # 假设你已经实现了一个名为SudokuDataset的数据集类


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs.view(-1, 10), labels.view(-1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(dataloader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SudokuDataset(...)  # 初始化你的数据集
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    model = SudokuResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {train_loss:.4f}")

    torch.save(model.state_dict(), "image_processing/model/sudoku_resnet.pth")


if __name__ == "__main__":
    main()
