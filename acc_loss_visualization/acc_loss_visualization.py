import re
import os
import matplotlib.pyplot as plt

LOG_DIR = os.path.join(os.path.dirname(__file__), "training_logs")

# 3 pairs: (pat_log, standalone_log, student_label, pat_label)
PAIRS = [
    ("pat_swint-resnet18_train.log", "standalone_resnet18_train.log", "ResNet-18", "PAT (Swin-T -> ResNet-18)"),
    ("pat_convnext-resmlp12_train.log", "standalone_resmlp_12_224_train.log", "ResMLP-12", "PAT (ConvNeXt -> ResMLP-12)"),
    ("pat_convnext-swinp_train.log", "swinp_train.log", "Swin Pico", "PAT (ConvNeXt -> Swin Pico)"),
]


def parse_pat_log(filepath):
    epochs = []
    losses = []
    accs = []

    current_epoch = None
    with open(filepath) as f:
        for line in f:
            m = re.search(r"Train:\s*(\d+)\s+\[\s*389/390\s+\(100%\)\]", line)
            if m:
                current_epoch = int(m.group(1))
                continue

            if "Test (EMA" not in line:
                m = re.search(
                    r"Test:\s+\[\s*78/78\].*?Loss:.*?\(([\d.]+)\).*?Acc@1:.*?\(\s*([\d.]+)\)",
                    line,
                )
                if m and current_epoch is not None:
                    losses.append(float(m.group(1)))
                    accs.append(float(m.group(2)))
                    epochs.append(current_epoch)
                    current_epoch = None

    return epochs, losses, accs


def parse_standalone_log(filepath):
    with open(filepath) as f:
        lines = f.readlines()

    last_start = 0
    for i, line in enumerate(lines):
        if re.match(r"Epoch\s+0/", line):
            last_start = i

    epochs, losses, accs = [], [], []
    for line in lines[last_start:]:
        m = re.search(
            r"Epoch\s+(\d+)/\d+.*?val_loss=([\d.]+)\s+val_acc1=([\d.]+)",
            line,
        )
        if m:
            epochs.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            accs.append(float(m.group(3)))

    return epochs, losses, accs


def main():
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    for row, (pat_file, standalone_file, label, pat_label) in enumerate(PAIRS):
        pat_path = os.path.join(LOG_DIR, pat_file)
        sa_path = os.path.join(LOG_DIR, standalone_file)

        pat_ep, pat_loss, pat_acc = parse_pat_log(pat_path)
        sa_ep, sa_loss, sa_acc = parse_standalone_log(sa_path)

        # Left column: Loss
        ax_loss = axes[row, 0]
        ax_loss.plot(pat_ep, pat_loss, label=pat_label)
        ax_loss.plot(sa_ep, sa_loss, label=f"Standalone {label}")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"{label} - Loss")
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)

        # Right column: Accuracy
        ax_acc = axes[row, 1]
        ax_acc.plot(pat_ep, pat_acc, label=pat_label)
        ax_acc.plot(sa_ep, sa_acc, label=f"Standalone {label}")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy (%)")
        ax_acc.set_title(f"{label} - Validation Accuracy")
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)

    fig.suptitle("PAT vs Standalone Training Comparison", fontsize=16, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "acc_loss_plots.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plots to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
