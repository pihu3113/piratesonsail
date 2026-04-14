"""
Train the Adaptive Maze MLP (4 -> 16 -> 8 -> 3) using synthetic gameplay data.

HOW TO RUN:
    pip install torch
    python train_model.py

OUTPUT:
    trained_weights.npz  — saved weights loaded by model.py at runtime
    training_plot.png    — loss curve (requires matplotlib)
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not found. Install it: pip install torch")
    exit(1)

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False


# ── 1. Synthetic Data Generation ─────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> tuple:
    """
    Simulate n_samples players and compute their 'ideal' difficulty factors.

    Input features (all in [0, 1], higher = worse performance):
        time_norm       — completion time / 300
        mistake_norm    — mistakes / 15
        inefficiency    — 1 - optimal_moves / actual_moves
        lives_lost_norm — lives lost / 3

    Target outputs (all in [0.1, 0.9], higher = harder next maze):
        size_factor       — fast + efficient players get bigger mazes
        rocks_factor      — accurate + careful players get more rocks
        complexity_factor — overall good players get complex layouts

    Ground truth logic:
        A player who performed WELL should get a HARDER next level.
        We encode this as: factor = 1 - normalized_struggle
        With added noise to simulate real player variability.
    """
    rng = np.random.default_rng(seed)

    # Raw inputs — sample from realistic distributions
    time_norm       = rng.beta(2, 3, n_samples)      # skewed toward faster times
    mistake_norm    = rng.beta(1.5, 4, n_samples)    # skewed toward fewer mistakes
    inefficiency    = rng.beta(2, 4, n_samples)      # skewed toward efficient paths
    lives_lost_norm = rng.beta(1, 4, n_samples)      # skewed toward not losing lives

    X = np.stack([time_norm, mistake_norm, inefficiency, lives_lost_norm], axis=1)

    # ── Ground truth targets ──────────────────────────────────────────────────

    # size_factor: driven by time + efficiency
    #   fast (low time_norm) + efficient (low inefficiency) → high size_factor
    time_perf       = 1.0 - time_norm          # 1 = fast, 0 = slow
    efficiency_perf = 1.0 - inefficiency       # 1 = efficient, 0 = wasteful
    size_raw = 0.6 * time_perf + 0.4 * efficiency_perf

    # rocks_factor: driven by accuracy + lives
    #   few mistakes (low mistake_norm) + few lives lost → high rocks_factor
    accuracy_perf   = 1.0 - mistake_norm
    lives_perf      = 1.0 - lives_lost_norm
    rocks_raw = 0.5 * accuracy_perf + 0.5 * lives_perf

    # complexity_factor: driven by all four metrics equally
    overall_perf  = 1.0 - (time_norm + mistake_norm + inefficiency + lives_lost_norm) / 4.0
    # Combo bonus: if BOTH time and efficiency are good, complexity goes up more
    combo_bonus   = np.where((time_perf > 0.6) & (efficiency_perf > 0.6), 0.15, 0.0)
    complexity_raw = overall_perf + combo_bonus

    # Add small Gaussian noise to simulate real variability
    noise = rng.normal(0, 0.05, (n_samples, 3))

    Y = np.stack([size_raw, rocks_raw, complexity_raw], axis=1) + noise

    # Clip targets to valid range [0.1, 0.9]
    Y = np.clip(Y, 0.1, 0.9).astype(np.float32)
    X = X.astype(np.float32)

    return X, Y


# ── 2. PyTorch MLP Definition ─────────────────────────────────────────────────

class MazeAdaptiveMLP(nn.Module):
    """
    MLP: 4 → 16 → 8 → 3
    Mirrors the architecture of the hand-crafted AdaptiveMazeModel in model.py
    but learns weights via backpropagation instead of manual assignment.

    Activations:
        Hidden layers : ReLU  (same as hand-crafted model)
        Output layer  : Sigmoid → maps to (0, 1), clipped to [0.1, 0.9] at inference
    """
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 3)
        self.relu    = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization — better than default for sigmoid outputs
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        # Bias of -2.2 on output layer → sigmoid(-2.2) ≈ 0.1
        # Ensures struggling player gets easiest maze by default
        nn.init.constant_(self.layer3.bias, -2.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1  = self.relu(self.layer1(x))
        h2  = self.relu(self.layer2(h1))
        out = self.sigmoid(self.layer3(h2))
        return out


# ── 3. Training Loop ──────────────────────────────────────────────────────────

def train(
    model: MazeAdaptiveMLP,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val:   np.ndarray,
    Y_val:   np.ndarray,
    epochs:      int   = 300,
    batch_size:  int   = 64,
    lr:          float = 1e-3,
    patience:    int   = 20,
) -> tuple[list, list]:
    """
    Train with MSE loss + Adam optimizer + early stopping.

    Returns:
        train_losses, val_losses  — one value per epoch for plotting
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    # Learning rate scheduler — halve LR if val loss plateaus for 10 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)

    # DataLoader for mini-batch training
    dataset    = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    X_val_t = torch.from_numpy(X_val)
    Y_val_t = torch.from_numpy(Y_val)

    train_losses, val_losses = [], []
    best_val_loss  = float("inf")
    best_weights   = None
    patience_count = 0

    for epoch in range(1, epochs + 1):

        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for X_batch, Y_batch in dataloader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        train_loss = epoch_loss / len(X_train)

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, Y_val_t).item()

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ── Early stopping ────────────────────────────────────────────────────
        if val_loss < best_val_loss - 1e-5:
            best_val_loss  = val_loss
            best_weights   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
                break

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:>4} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    # Restore best weights
    if best_weights:
        model.load_state_dict(best_weights)

    return train_losses, val_losses


# ── 4. Evaluation ─────────────────────────────────────────────────────────────

def evaluate(model: MazeAdaptiveMLP, X_test: np.ndarray, Y_test: np.ndarray):
    """Print per-output MAE and a few sample predictions vs ground truth."""
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_test)).numpy()

    mae_per_output = np.mean(np.abs(preds - Y_test), axis=0)
    print(f"\n  Test MAE — size_factor: {mae_per_output[0]:.4f} | "
          f"rocks_factor: {mae_per_output[1]:.4f} | "
          f"complexity_factor: {mae_per_output[2]:.4f}")
    print(f"  Overall test MAE: {mae_per_output.mean():.4f}")

    print("\n  Sample predictions (first 5 test examples):")
    print(f"  {'Input (t,m,i,l)':<30} {'Predicted':<35} {'Ground Truth'}")
    print("  " + "-" * 85)
    for i in range(5):
        inp  = X_test[i]
        pred = preds[i]
        true = Y_test[i]
        print(f"  {str(np.round(inp, 2)):<30} "
              f"{str(np.round(pred, 3)):<35} "
              f"{str(np.round(true, 3))}")


# ── 5. Save Weights for model.py ─────────────────────────────────────────────

def save_weights(model: MazeAdaptiveMLP, path: str = "trained_weights.npz"):
    """
    Extract weights from PyTorch model and save as .npz so that
    model.py can load them using only numpy (no PyTorch dependency at runtime).
    """
    sd = model.state_dict()
    np.savez(
        path,
        W1 = sd["layer1.weight"].numpy().T,   # shape (4, 16)
        b1 = sd["layer1.bias"].numpy(),        # shape (16,)
        W2 = sd["layer2.weight"].numpy().T,   # shape (16, 8)
        b2 = sd["layer2.bias"].numpy(),        # shape (8,)
        W3 = sd["layer3.weight"].numpy().T,   # shape (8, 3)
        b3 = sd["layer3.bias"].numpy(),        # shape (3,)
    )
    print(f"\n  Weights saved to '{path}'")
    print("  model.py will automatically load these on next startup.")


# ── 6. Plot Loss Curves ───────────────────────────────────────────────────────

def plot_losses(train_losses: list, val_losses: list, path: str = "training_plot.png"):
    if not PLOT_AVAILABLE:
        print("  matplotlib not installed — skipping plot (pip install matplotlib)")
        return

    plt.figure(figsize=(9, 4))
    plt.plot(train_losses, label="Train Loss", color="#2196F3", linewidth=1.5)
    plt.plot(val_losses,   label="Val Loss",   color="#FF5722", linewidth=1.5, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MLP Training: Adaptive Maze Difficulty Model (4→16→8→3)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"  Loss plot saved to '{path}'")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Adaptive Maze MLP — Training Script")
    print("=" * 60)

    # 1. Generate data
    print("\n[1/5] Generating synthetic gameplay data...")
    X, Y = generate_synthetic_data(n_samples=5000)
    print(f"  Total samples : {len(X)}")
    print(f"  Input shape   : {X.shape}  (time, mistakes, inefficiency, lives_lost)")
    print(f"  Output shape  : {Y.shape}  (size_factor, rocks_factor, complexity_factor)")
    print(f"  Output ranges : min={Y.min(axis=0).round(3)}, max={Y.max(axis=0).round(3)}")

    # 2. Train/val/test split — 70 / 15 / 15
    print("\n[2/5] Splitting data (70% train / 15% val / 15% test)...")
    n        = len(X)
    n_train  = int(0.70 * n)
    n_val    = int(0.15 * n)
    idx      = np.random.default_rng(0).permutation(n)
    X_train, Y_train = X[idx[:n_train]],          Y[idx[:n_train]]
    X_val,   Y_val   = X[idx[n_train:n_train+n_val]], Y[idx[n_train:n_train+n_val]]
    X_test,  Y_test  = X[idx[n_train+n_val:]],    Y[idx[n_train+n_val:]]
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 3. Build model
    print("\n[3/5] Building MLP (4 → 16 → 8 → 3)...")
    model = MazeAdaptiveMLP()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total trainable parameters: {total_params}")
    print(f"  Architecture: Linear(4,16)->ReLU -> Linear(16,8)->ReLU -> Linear(8,3)->Sigmoid")

    # 4. Train
    print("\n[4/5] Training (MSE loss, Adam optimizer, early stopping)...")
    train_losses, val_losses = train(
        model, X_train, Y_train, X_val, Y_val,
        epochs=300, batch_size=64, lr=1e-3, patience=20
    )
    print(f"  Training complete. Final val loss: {val_losses[-1]:.6f}")

    # 5. Evaluate
    print("\n[5/5] Evaluating on test set...")
    evaluate(model, X_test, Y_test)

    # Save weights and plot
    save_weights(model, "trained_weights.npz")
    plot_losses(train_losses, val_losses, "training_plot.png")

    print("\n" + "=" * 60)
    print("  Done! Run the game — model.py will use trained weights.")
    print("=" * 60)
