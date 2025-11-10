import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from .dataset import DummyPlays
from .model import SimplePlayCNN

def train():
    ds = DummyPlays()
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    net = SimplePlayCNN()
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=1e-3)

    for epoch in range(3):
        total, correct, loss_sum = 0, 0, 0.0
        for x,y in dl:
            opt.zero_grad()
            logits = net(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss)
            pred = logits.argmax(1)
            total += y.numel()
            correct += int((pred==y).sum())
        print(f"epoch {epoch+1} loss={loss_sum/len(dl):.4f} acc={correct/total:.3f}")
