"""
Fine-tune the MLP using real player gameplay data collected by app.py.

HOW IT WORKS:
    1. Reads gameplay_data.json  — real player sessions
    2. Converts next_level_performance into training labels
    3. Mixes real data with synthetic data (avoids forgetting when real data is small)
    4. Fine-tunes from current trained_weights.npz
    5. Saves updated weights back to trained_weights.npz

HOW TO RUN:
    py -3.13 retrain.py

Run this periodically as more players complete the game (e.g. every 10 new sessions).

LABEL LOGIC (delayed feedback):
    The model predicted difficulty factors for level N+1 based on level N stats.
    Now we see how the player ACTUALLY performed on level N+1.

    If player was fast on level N+1  → level N+1 was too easy → size_factor should have been HIGHER
    If player was slow on level N+1  → level N+1 was too hard → size_factor should have been LOWER

    size_factor_label       = 1 - time_norm_next        (fast next → should have been bigger)
    rocks_factor_label      = 1 - mistake_norm_next     (few mistakes → should have had more rocks)
    complexity_factor_label = 1 - inefficiency_next     (efficient next → should have been more complex)
"""

import json
import os
import numpy as np

GAMEPLAY_DATA_FILE  = os.path.join(os.path.dirname(__file__), "gameplay_data.json")
TRAINED_WEIGHTS     = os.path.join(os.path.dirname(__file__), "trained_weights.npz")
MIN_REAL_SAMPLES    = 5      # retrain only if we have at least this many real samples
REAL_DATA_RATIO     = 0.30   # 30% real data, 70% synthetic in mixed dataset


# ── Activations ───────────────────────────────────────────────────────────────

def relu(x):         return np.maximum(0.0, x)
def relu_grad(x):    return (x > 0).astype(np.float64)
def sigmoid(x):      return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))
def sigmoid_grad(x): s = sigmoid(x); return s * (1.0 - s)
def mse_loss(p, y):  return np.mean((p - y) ** 2)
def mse_grad(p, y):  return 2.0 * (p - y) / p.size


# ── Load real gameplay data ───────────────────────────────────────────────────

def load_real_data(path: str) -> tuple:
    """
    Read gameplay_data.json and convert to (X, Y) arrays.

    X: input features from level N  (what player did)
    Y: labels derived from level N+1 performance (delayed feedback)
    """
    with open(path) as f:
        records = json.load(f)

    X_list, Y_list = [], []

    for r in records:
        inp  = r["input"]
        nxt  = r["next_level_performance"]

        x = [
            inp["time_norm"],
            inp["mistake_norm"],
            inp["inefficiency"],
            inp["lives_lost_norm"],
        ]

        # Label: if next level was easy → difficulty should have been higher
        size_f       = 1.0 - nxt["time_norm"]          # fast player → bigger maze needed
        rocks_f      = 1.0 - nxt["mistake_norm"]       # few mistakes → more rocks needed
        complexity_f = 1.0 - nxt["inefficiency"]       # efficient → more complex needed

        y = [
            np.clip(size_f,       0.1, 0.9),
            np.clip(rocks_f,      0.1, 0.9),
            np.clip(complexity_f, 0.1, 0.9),
        ]

        X_list.append(x)
        Y_list.append(y)

    return np.array(X_list, dtype=np.float64), np.array(Y_list, dtype=np.float64)


# ── Synthetic data (same logic as train_model.py) ────────────────────────────

def generate_synthetic_data(n_samples: int = 3000, seed: int = 99) -> tuple:
    rng = np.random.default_rng(seed)

    time_norm       = rng.beta(2, 3, n_samples)
    mistake_norm    = rng.beta(1.5, 4, n_samples)
    inefficiency    = rng.beta(2, 4, n_samples)
    lives_lost_norm = rng.beta(1, 4, n_samples)

    X = np.stack([time_norm, mistake_norm, inefficiency, lives_lost_norm], axis=1)

    time_perf       = 1.0 - time_norm
    efficiency_perf = 1.0 - inefficiency
    accuracy_perf   = 1.0 - mistake_norm
    lives_perf      = 1.0 - lives_lost_norm

    size_raw       = 0.6 * time_perf + 0.4 * efficiency_perf
    rocks_raw      = 0.5 * accuracy_perf + 0.5 * lives_perf
    overall        = (time_perf + accuracy_perf + efficiency_perf + lives_perf) / 4.0
    combo_bonus    = np.where((time_perf > 0.6) & (efficiency_perf > 0.6), 0.15, 0.0)
    complexity_raw = overall + combo_bonus

    noise = rng.normal(0, 0.04, (n_samples, 3))
    Y = np.stack([size_raw, rocks_raw, complexity_raw], axis=1) + noise
    Y = np.clip(Y, 0.1, 0.9)

    return X.astype(np.float64), Y.astype(np.float64)


# ── MLP (weights loaded from file) ───────────────────────────────────────────

class MLP:
    def __init__(self, weights_path: str):
        data     = np.load(weights_path)
        self.W1  = data["W1"].astype(np.float64)
        self.b1  = data["b1"].astype(np.float64)
        self.W2  = data["W2"].astype(np.float64)
        self.b2  = data["b2"].astype(np.float64)
        self.W3  = data["W3"].astype(np.float64)
        self.b3  = data["b3"].astype(np.float64)
        self._init_adam()

    def _init_adam(self):
        self.adam = {}
        for name in ["W1","b1","W2","b2","W3","b3"]:
            p = getattr(self, name)
            self.adam[name] = {"m": np.zeros_like(p), "v": np.zeros_like(p)}
        self.adam_t = 0

    def forward(self, X):
        z1 = X  @ self.W1 + self.b1;  a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2;  a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3;  a3 = sigmoid(z3)
        return a3, (X, z1, a1, z2, a2, z3)

    def backward(self, a3, Y, cache):
        X, z1, a1, z2, a2, z3 = cache
        b = X.shape[0]

        dz3 = mse_grad(a3, Y) * sigmoid_grad(z3)
        dW3 = a2.T @ dz3 / b;   db3 = dz3.mean(0)

        dz2 = (dz3 @ self.W3.T) * relu_grad(z2)
        dW2 = a1.T @ dz2 / b;   db2 = dz2.mean(0)

        dz1 = (dz2 @ self.W2.T) * relu_grad(z1)
        dW1 = X.T  @ dz1 / b;   db1 = dz1.mean(0)

        return {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2,"W3":dW3,"b3":db3}

    def adam_step(self, grads, lr=5e-4):
        self.adam_t += 1
        t = self.adam_t
        for name in ["W1","b1","W2","b2","W3","b3"]:
            g = grads[name]
            m = 0.9   * self.adam[name]["m"] + 0.1   * g
            v = 0.999 * self.adam[name]["v"] + 0.001 * g**2
            self.adam[name].update({"m": m, "v": v})
            m_hat = m / (1 - 0.9   ** t)
            v_hat = v / (1 - 0.999 ** t)
            setattr(self, name, getattr(self, name) - lr * m_hat / (np.sqrt(v_hat) + 1e-8))

    def predict(self, X):
        out, _ = self.forward(X)
        return np.clip(out, 0.1, 0.9)

    def save(self, path: str):
        np.savez(path, W1=self.W1, b1=self.b1,
                       W2=self.W2, b2=self.b2,
                       W3=self.W3, b3=self.b3)


# ── Fine-tuning loop ──────────────────────────────────────────────────────────

def fine_tune(model, X, Y, epochs=100, batch_size=32, lr=5e-4, patience=15):
    """
    Fine-tune with a lower LR than initial training to avoid forgetting.
    Uses same Adam + early stopping approach.
    """
    rng = np.random.default_rng(7)
    n   = len(X)

    # 80/20 train/val split
    idx    = rng.permutation(n)
    split  = int(0.8 * n)
    X_tr, Y_tr = X[idx[:split]],  Y[idx[:split]]
    X_val, Y_val = X[idx[split:]], Y[idx[split:]]

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        idx2 = rng.permutation(len(X_tr))
        X_s, Y_s = X_tr[idx2], Y_tr[idx2]

        for start in range(0, len(X_tr), batch_size):
            Xb, Yb      = X_s[start:start+batch_size], Y_s[start:start+batch_size]
            pred, cache = model.forward(Xb)
            grads       = model.backward(pred, Yb, cache)
            model.adam_step(grads, lr=lr)

        val_pred, _ = model.forward(X_val)
        val_loss    = mse_loss(val_pred, Y_val)

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_state = {k: getattr(model, k).copy()
                          for k in ["W1","b1","W2","b2","W3","b3"]}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best val: {best_val:.6f})")
                break

        if epoch % 25 == 0:
            print(f"  Epoch {epoch:>4} | Val Loss: {val_loss:.6f}")

    if best_state:
        for k, v in best_state.items():
            setattr(model, k, v)

    return best_val


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Adaptive Maze MLP — Retrain with Real Player Data")
    print("=" * 60)

    # 1. Check real data exists
    if not os.path.exists(GAMEPLAY_DATA_FILE):
        print(f"\n  gameplay_data.json not found.")
        print(f"  Play the game first — data is collected automatically.")
        exit(0)

    print(f"\n[1/4] Loading real gameplay data from '{GAMEPLAY_DATA_FILE}'...")
    X_real, Y_real = load_real_data(GAMEPLAY_DATA_FILE)
    print(f"  Real samples: {len(X_real)}")

    if len(X_real) < MIN_REAL_SAMPLES:
        print(f"  Only {len(X_real)} samples — need at least {MIN_REAL_SAMPLES} to retrain.")
        print(f"  Play more games and run this again.")
        exit(0)

    # 2. Generate synthetic data and mix
    print(f"\n[2/4] Mixing with synthetic data (ratio: {int(REAL_DATA_RATIO*100)}% real)...")
    n_synthetic   = int(len(X_real) * (1 - REAL_DATA_RATIO) / REAL_DATA_RATIO)
    X_syn, Y_syn  = generate_synthetic_data(n_samples=max(n_synthetic, 500))
    X_mixed       = np.vstack([X_real, X_syn])
    Y_mixed       = np.vstack([Y_real, Y_syn])
    print(f"  Real: {len(X_real)} | Synthetic: {len(X_syn)} | Total: {len(X_mixed)}")

    # Shuffle
    idx     = np.random.default_rng(0).permutation(len(X_mixed))
    X_mixed = X_mixed[idx]
    Y_mixed = Y_mixed[idx]

    # 3. Load current model and fine-tune
    if not os.path.exists(TRAINED_WEIGHTS):
        print(f"\n  trained_weights.npz not found. Run train_model.py first.")
        exit(1)

    print(f"\n[3/4] Fine-tuning from '{TRAINED_WEIGHTS}'...")
    print(f"  Using lower LR (5e-4) to preserve existing knowledge...")
    model    = MLP(TRAINED_WEIGHTS)
    best_val = fine_tune(model, X_mixed, Y_mixed,
                         epochs=150, batch_size=32, lr=5e-4, patience=15)
    print(f"  Fine-tuning complete. Best val loss: {best_val:.6f}")

    # 4. Save
    print(f"\n[4/4] Saving updated weights...")
    model.save(TRAINED_WEIGHTS)
    print(f"  Saved → '{TRAINED_WEIGHTS}'")
    print(f"  Restart the game server to use updated weights.")

    print("\n" + "=" * 60)
    print(f"  Done! Model updated with {len(X_real)} real player sessions.")
    print("=" * 60)
