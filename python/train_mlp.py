"""SUPERSEDED -- digit MNIST, not the ASL model this project ships.

Kept as the original bring-up script. The deployed network is trained by
train_asl_v2.py on Sign Language MNIST (24 classes).

Originally: stage 1, train the float32 reference MLP (784 -> 64 -> 10).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(0)

HIDDEN = 64
EPOCHS = 8
BATCH = 128
LR = 1e-3


class MLP(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def get_loaders():
    # No normalization on purpose: hardware sees a raw 0-255 pixel, so we
    # keep the training domain as a simple scaling of that.
    tf = transforms.ToTensor()
    train = datasets.MNIST("./data", train=True, download=True, transform=tf)
    test = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=BATCH, shuffle=True),
            DataLoader(test, batch_size=1000))


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    for x, y in loader:
        correct += (model(x).argmax(dim=1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    train_loader, test_loader = get_loaders()
    model = MLP()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossfn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            opt.zero_grad()
            lossfn(model(x), y).backward()
            opt.step()
        print(f"epoch {epoch:2d}  test acc {evaluate(model, test_loader)*100:.2f}%")

    acc = evaluate(model, test_loader)
    print(f"\nFP32 baseline accuracy: {acc*100:.2f}%   <-- record this number")

    torch.save(model.state_dict(), "mlp_fp32.pt")
    print("saved mlp_fp32.pt")

    for name, p in model.named_parameters():
        print(f"  {name:12s} shape {tuple(p.shape)}  "
              f"min {p.min():+.4f}  max {p.max():+.4f}")


if __name__ == "__main__":
    main()
