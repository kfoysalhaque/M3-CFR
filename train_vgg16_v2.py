import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import vgg16_bn

from m3cfr_datagen import M3CFRDataset, create_train_test_split


def build_vgg16(num_classes=10, in_channels=10):
    model = vgg16_bn(weights=None)

    model.features[0] = nn.Conv2d(
        in_channels=in_channels,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )

    model.classifier[6] = nn.Linear(4096, num_classes)

    return model


def sanity_check_loader(loader):
    x, y = next(iter(loader))

    print("\n===== Data Sanity Check =====")
    print(f"Input shape: {x.shape}")
    print(f"Label shape: {y.shape}")
    print(f"Input min: {x.min().item():.6f}")
    print(f"Input max: {x.max().item():.6f}")
    print(f"Input mean: {x.mean().item():.6f}")
    print(f"Input std: {x.std().item():.6f}")
    print(f"Labels in batch: {torch.unique(y)}")
    print(f"Label count: {torch.bincount(y, minlength=10)}")
    print("=============================\n")


def overfit_one_batch(model, loader, criterion, device, lr=1e-4, steps=200):
    print("\n===== One-Batch Overfit Test =====")

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    x, y = next(iter(loader))
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)

    for step in range(steps):
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        acc = (outputs.argmax(1) == y).float().mean().item()

        if step % 20 == 0 or step == steps - 1:
            print(
                f"Step [{step+1}/{steps}] "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {acc*100:.2f}%"
            )

    print("==================================\n")


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    num_epochs,
    print_every=10
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(x)
            loss = criterion(outputs, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        correct += (outputs.argmax(1) == y).sum().item()
        total += y.size(0)

        if (batch_idx + 1) % print_every == 0 or (batch_idx + 1) == len(loader):
            running_loss = total_loss / total
            running_acc = correct / total

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Step [{batch_idx+1}/{len(loader)}] "
                f"Loss: {running_loss:.4f} | "
                f"Acc: {running_acc*100:.2f}%"
            )

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(x)
                loss = criterion(outputs, y)

            total_loss += loss.item() * y.size(0)
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

    return total_loss / total, correct / total


def main():
    root_dir = "./"

    env = "lab"
    subject = "sub-3"

    gestures = [
        "Finger-Crossed", "Fist", "half-palm", "okay", "pinch",
        "pointing", "tap-fingers", "thumbs-down", "thumbs-up", "victory"
    ]

    full_dataset = M3CFRDataset(
        root_dir=root_dir,
        env=env,
        subject=subject,
        gestures=gestures,
        window_size=10,
        stride=5
    )

    train_dataset, test_dataset = create_train_test_split(
        full_dataset,
        train_ratio=0.8,
        seed=42
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Environment: {env}")
    print(f"Subject: {subject}")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Testing batches: {len(test_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sanity_check_loader(train_loader)

    model = build_vgg16(num_classes=10, in_channels=10).to(device)

    criterion = nn.CrossEntropyLoss()

    # Run this first. If it cannot overfit one batch, full training is meaningless.
    run_overfit_test = True

    if run_overfit_test:
        overfit_one_batch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            lr=1e-4,
            steps=200
        )

        # Reinitialize model after overfit test
        model = build_vgg16(num_classes=10, in_channels=10).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    num_epochs = 50
    best_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
            print_every=10
        )

        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device
        )

        print(
            f"\nEpoch [{epoch+1}/{num_epochs}] Summary | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc*100:.2f}%\n"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            save_name = f"best_vgg16_bn_{env}_{subject}.pth"
            torch.save(model.state_dict(), save_name)
            print(f"Saved best model: {save_name} | Accuracy: {best_acc*100:.2f}%\n")

    print(f"Best Test Accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
