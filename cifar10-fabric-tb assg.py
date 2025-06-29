import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision.datasets as datasets
import torchvision.transforms.v2 as v2

from einops.layers.torch import Reduce
from torchmetrics.classification import Accuracy

from lightning.fabric import Fabric, seed_everything
from lightning.fabric.loggers import TensorBoardLogger

def main():
    parser = argparse.ArgumentParser(description="multi-gpu image classification")
    # Fabric args
    parser.add_argument("--accelerator", default="gpu", type=str)
    parser.add_argument("--devices",     default=2,   type=int)
    parser.add_argument("--num_nodes",   default=1,   type=int)
    parser.add_argument("--strategy",    default="ddp",type=str)
    parser.add_argument("--precision",   default="bf16-mixed", type=str)
    parser.add_argument("--seed",        default=1,   type=int)
    # training args
    parser.add_argument("--batch-size",  default=64,  type=int)
    parser.add_argument("--num-workers", default=4,   type=int)
    parser.add_argument("--epochs",      default=10,  type=int)
    parser.add_argument("--lr",          default=1e-3,type=float)
    parser.add_argument("--gamma",       default=0.7, type=float)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    # 1) TensorBoardLogger writes events into ./logs/tb/cifar10-multi-gpu
    tb_logger = TensorBoardLogger(root_dir="./logs/tb", name="cifar10-multi-gpu")
    fabric   = Fabric(accelerator=args.accelerator,
                      devices=args.devices,
                      num_nodes=args.num_nodes,
                      strategy=args.strategy,
                      precision=args.precision,
                      loggers=[tb_logger])
    fabric.launch()
    seed_everything(args.seed)

    # 2) Prepare data (only rank zero downloads)
    with fabric.rank_zero_first(local=False):
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
        download = not Path("./data/cifar-10-batches-py").exists()
        train_ds = datasets.CIFAR10("./data", train=True,  download=download, transform=transform)
        test_ds  = datasets.CIFAR10("./data", train=False, download=download, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True)

    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    # 3) Model, loss, optimizer, scheduler
    num_classes = 10
    with fabric.init_module():
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),   nn.BatchNorm2d(64), nn.ReLU(),
            Reduce("b c (h 2) (w 2) -> b (c h w)", "max"),
            nn.Linear(64*8*8,128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    model, optimizer = fabric.setup(model, optimizer)

    # 4) Metrics (moved to device)
    train_acc = Accuracy(task="multiclass", num_classes=num_classes).to(fabric.device)
    test_acc  = Accuracy(task="multiclass", num_classes=num_classes).to(fabric.device)

    # 5) Optionally dry-run
    for epoch in range(args.epochs):
        # ----- TRAIN -----
        model.train()
        total_train_loss = 0.0
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            fabric.backward(loss)
            optimizer.step()

            total_train_loss += loss.item()
            train_acc(out, labels)

        # average across all GPUs
        avg_train_loss = fabric.all_gather(total_train_loss).sum() / len(train_loader.dataset)
        avg_train_acc  = train_acc.compute()

        # ----- VALIDATE -----
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for imgs, labels in test_loader:
                out = model(imgs)
                total_test_loss += loss_fn(out, labels).item()
                test_acc(out, labels)

        avg_test_loss = fabric.all_gather(total_test_loss).sum() / len(test_loader.dataset)
        avg_test_acc  = test_acc.compute()

        # ----- LOG to TensorBoard -----
        fabric.log_dict({
            "train/loss": avg_train_loss,
            "train/accuracy": 100 * avg_train_acc,
            "test/loss":  avg_test_loss,
            "test/accuracy":   100 * avg_test_acc
        }, step=epoch)

        # print & step
        fabric.print(f"Epoch {epoch:2d} — "
                     f"Train: loss={avg_train_loss:.4f}, acc={100*avg_train_acc:.1f}% | "
                     f"Test:  loss={avg_test_loss:.4f}, acc={100*avg_test_acc:.1f}%")
        scheduler.step()

        # reset metrics
        train_acc.reset(); test_acc.reset()

        if args.dry_run:
            break

if __name__ == "__main__":
    main()
