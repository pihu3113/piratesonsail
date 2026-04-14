"""
Adaptive Maze Difficulty Model — MLP (4 → 16 → 8 → 3)

At startup, tries to load weights from 'trained_weights.npz' (produced by
train_model.py).  If the file is absent, falls back to the original
hand-crafted weights so the game always runs.

Run once to train:
    pip install torch
    python train_model.py

Inputs  (4 features, all normalised to [0, 1], higher = worse performance):
    time_norm       — completion time / 300 s
    mistake_norm    — mistakes / 15
    inefficiency    — 1 - optimal_moves / actual_moves
    lives_lost_norm — lives lost / 3

Outputs (3 INDEPENDENT difficulty factors, each in [0.1, 0.9]):
    size_factor       — driven by time + efficiency signals
    rocks_factor      — driven by mistakes + lives signals
    complexity_factor — driven by all signals equally

Adaptation logic (higher factor = harder next maze):
    Fast + efficient player   → high size_factor    → bigger maze
    Accurate + careful player → high rocks_factor   → more obstacles
    Overall good player       → high complexity_factor → more complex layout

Signal flow (three independent pathways through the network):
    H1 neurons:
        0  : time inverter            (fires high when time  is LOW = fast)
        1  : mistakes inverter        (fires high when mistakes is LOW = accurate)
        2  : efficiency inverter      (fires high when inefficiency is LOW = efficient)
        3  : lives inverter           (fires high when lives_lost is LOW = careful)
        4-6: overall performance (uniform, mistake-weighted, time-weighted)
        7  : overall performance (efficiency-weighted)
        8  : time struggle            (fires high when time is HIGH = slow)
        9  : mistakes struggle        (fires high when mistakes is HIGH)
        10 : efficiency struggle      (fires high when inefficiency is HIGH)
        11 : lives struggle           (fires high when lives_lost is HIGH)
        12 : fast + efficient combo   (fires when both time and efficiency are good)
        13 : accurate + safe combo    (fires when both mistakes and lives are good)
        14 : slow + many-mistakes combo (fires when both are bad)
        15 : inefficient + lives-lost combo (fires when both are bad)

    H2 neurons (specialised pathways):
        0-1 : TIME pathway   — receives time-specific H1 signals
        2-3 : MISTAKES pathway — receives mistake-specific H1 signals
        4-5 : EFFICIENCY pathway — receives efficiency-specific H1 signals
        6-7 : LIVES pathway  — receives lives-specific H1 signals

    H3 (outputs):
        size_factor       = f(H2 time-pathway, H2 efficiency-pathway)
        rocks_factor      = f(H2 mistakes-pathway, H2 lives-pathway)
        complexity_factor = f(all H2 pathways equally)
"""

import os
import numpy as np

TRAINED_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "trained_weights.npz")


class AdaptiveMazeModel:

    def __init__(self):
        if os.path.exists(TRAINED_WEIGHTS_PATH):
            self._load_trained_weights(TRAINED_WEIGHTS_PATH)
            return

        # ── Fallback: hand-crafted weights (used when trained_weights.npz absent) ──
        print("[model] trained_weights.npz not found — using hand-crafted weights.")
        print("[model] Run 'python train_model.py' to train the model.")
        self._init_handcrafted_weights()

    def _load_trained_weights(self, path: str):
        """Load weights saved by train_model.py (numpy .npz format)."""
        data     = np.load(path)
        self.W1  = data["W1"].astype(np.float64)   # (4, 16)
        self.b1  = data["b1"].astype(np.float64)   # (16,)
        self.W2  = data["W2"].astype(np.float64)   # (16, 8)
        self.b2  = data["b2"].astype(np.float64)   # (8,)
        self.W3  = data["W3"].astype(np.float64)   # (8, 3)
        self.b3  = data["b3"].astype(np.float64)   # (3,)
        print(f"[model] Loaded trained weights from '{path}'")

    def _init_handcrafted_weights(self):
        rng = np.random.default_rng(seed=42)

        # ── Layer 1  (4 → 16)  hand-crafted weights ──────────────────────────
        self.W1 = rng.normal(0, 0.06, (4, 16))   # small random base
        self.b1 = np.zeros(16)

        # --- Neurons 0-3: individual metric inverters ---
        # Diagonal strong-negative: neuron i fires HIGH when input i is LOW.
        for i in range(4):
            self.W1[i, i] = -3.5
        self.b1[0:4] = 2.2

        # --- Neurons 4-7: overall performance (uniform + per-metric emphasis) ---
        self.W1[:, 4] = [-1.8, -1.8, -1.8, -1.8]         # uniform overall
        self.W1[:, 5] = [-0.8, -3.2, -0.8, -0.8]          # mistake-emphasis
        self.W1[:, 6] = [-3.2, -0.8, -0.8, -0.8]          # time-emphasis
        self.W1[:, 7] = [-0.8, -0.8, -3.2, -1.2]          # efficiency-emphasis
        self.b1[4:8] = 4.0

        # --- Neurons 8-11: struggle detectors (fires HIGH when metric is HIGH) ---
        for i in range(4):
            self.W1[i, 8 + i] = 3.5
        self.b1[8:12] = -2.2

        # --- Neuron 12: fast + efficient combo (fires when time AND efficiency both good) ---
        self.W1[0, 12] = -2.5   # time contribution
        self.W1[2, 12] = -2.5   # efficiency contribution
        self.b1[12] = 3.5

        # --- Neuron 13: accurate + safe combo (fires when mistakes AND lives both good) ---
        self.W1[1, 13] = -2.5   # mistakes contribution
        self.W1[3, 13] = -2.5   # lives contribution
        self.b1[13] = 3.5

        # --- Neuron 14: slow + inaccurate combo (fires when time AND mistakes both bad) ---
        self.W1[0, 14] = 2.5
        self.W1[1, 14] = 2.5
        self.b1[14] = -3.5

        # --- Neuron 15: inefficient + lives-lost combo (fires when both bad) ---
        self.W1[2, 15] = 2.5
        self.W1[3, 15] = 2.5
        self.b1[15] = -3.5

        # ── Layer 2  (16 → 8)  SPECIALISED PATHWAYS ─────────────────────────
        # Each pair of H2 neurons receives ONLY the H1 signals relevant to its dimension.
        # Start from zero so pathways are truly isolated.
        self.W2 = np.zeros((16, 8))
        self.b2 = np.zeros(8)

        # --- H2[0,1]: TIME pathway ---
        # Receives: N0 (time inverter), N6 (time-weighted overall), N8 (time struggle), N12 (combo)
        for j in (0, 1):
            self.W2[0, j]  =  1.6    # time inverter → fires when fast
            self.W2[6, j]  =  1.0    # time-weighted overall performance
            self.W2[8, j]  = -1.6    # time struggle suppresses
            self.W2[12, j] =  0.8    # fast+efficient combo boosts
            self.W2[14, j] = -0.8    # slow+inaccurate suppresses
        self.b2[0:2] = 0.2

        # --- H2[2,3]: MISTAKES pathway ---
        # Receives: N1 (mistakes inverter), N5 (mistake-weighted overall), N9 (mistakes struggle)
        for j in (2, 3):
            self.W2[1, j]  =  1.6    # mistakes inverter → fires when accurate
            self.W2[5, j]  =  1.0    # mistake-weighted overall performance
            self.W2[9, j]  = -1.6    # mistakes struggle suppresses
            self.W2[13, j] =  0.8    # accurate+safe combo boosts
            self.W2[14, j] = -0.8    # slow+inaccurate suppresses
        self.b2[2:4] = 0.2

        # --- H2[4,5]: EFFICIENCY pathway ---
        # Receives: N2 (efficiency inverter), N7 (efficiency-weighted overall), N10 (efficiency struggle)
        for j in (4, 5):
            self.W2[2, j]  =  1.6    # efficiency inverter → fires when efficient
            self.W2[7, j]  =  1.0    # efficiency-weighted overall performance
            self.W2[10, j] = -1.6    # efficiency struggle suppresses
            self.W2[12, j] =  0.8    # fast+efficient combo boosts
            self.W2[15, j] = -0.8    # inefficient+lives-lost suppresses
        self.b2[4:6] = 0.2

        # --- H2[6,7]: LIVES pathway ---
        # Receives: N3 (lives inverter), N11 (lives struggle), N13 (accurate+safe combo)
        for j in (6, 7):
            self.W2[3, j]  =  1.6    # lives inverter → fires when careful
            self.W2[11, j] = -1.6    # lives struggle suppresses
            self.W2[13, j] =  0.8    # accurate+safe combo boosts
            self.W2[15, j] = -0.8    # inefficient+lives-lost suppresses
            self.W2[4, j]  =  0.5    # mild overall-performance boost
        self.b2[6:8] = 0.2

        # ── Layer 3  (8 → 3)  SPECIALISED OUTPUTS ───────────────────────────
        #
        # size_factor       ← TIME pathway ONLY   (fast player → bigger maze)
        # rocks_factor      ← MISTAKES + LIVES    (accurate/careful → more rocks)
        # complexity_factor ← EFFICIENCY + overall (efficient → more complex paths)
        #
        # b3 = -2.2 so sigmoid(-2.2) ≈ 0.1 when all H2 inputs are zero
        # (ensures a struggling player gets the easiest possible maze variant)
        self.W3 = np.zeros((8, 3))
        self.b3 = np.array([-2.2, -2.2, -2.2])

        # size_factor: ONLY time pathway (H2[0], H2[1])
        self.W3[0, 0] = 0.35   # time pathway H2[0]
        self.W3[1, 0] = 0.35   # time pathway H2[1]

        # rocks_factor: mistakes (H2[2,3]) + lives (H2[6,7])
        self.W3[2, 1] = 0.35   # mistakes pathway H2[2]
        self.W3[3, 1] = 0.35   # mistakes pathway H2[3]
        self.W3[6, 1] = 0.35   # lives pathway H2[6]
        self.W3[7, 1] = 0.35   # lives pathway H2[7]

        # complexity_factor: efficiency (H2[4,5]) + mistakes + lives (broader signal)
        self.W3[4, 2] = 0.30   # efficiency pathway H2[4]
        self.W3[5, 2] = 0.30   # efficiency pathway H2[5]
        self.W3[2, 2] = 0.15   # mistakes pathway (contribution to complexity)
        self.W3[3, 2] = 0.15
        self.W3[6, 2] = 0.15   # lives pathway (contribution to complexity)
        self.W3[7, 2] = 0.15

    # ── Activations ──────────────────────────────────────────────────────────

    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1  = self.relu(x @ self.W1 + self.b1)
        h2  = self.relu(h1 @ self.W2 + self.b2)
        out = self.sigmoid(h2 @ self.W3 + self.b3)
        return out

    # ── Public API ───────────────────────────────────────────────────────────

    def predict_difficulty(self, time_taken: float, mistakes: int,
                           moves: int, optimal_moves: int,
                           lives_left: int = 3) -> dict:
        """
        Run one forward pass and return per-dimension difficulty factors.

        Returns
        -------
        dict:
            size_factor       in [0.1, 0.9] — higher = bigger maze
            rocks_factor      in [0.1, 0.9] — higher = more obstacles
            complexity_factor in [0.1, 0.9] — higher = more complex layout
        """
        time_norm       = min(time_taken / 300.0, 1.0)
        mistake_norm    = min(mistakes   / 15.0,  1.0)
        efficiency      = min(optimal_moves / max(moves, 1), 1.0)
        inefficiency    = 1.0 - efficiency
        lives_lost_norm = (3 - max(0, min(3, lives_left))) / 3.0

        x   = np.array([time_norm, mistake_norm, inefficiency, lives_lost_norm])
        out = self.forward(x)
        out = np.clip(out, 0.1, 0.9)

        return {
            "size_factor":       float(out[0]),
            "rocks_factor":      float(out[1]),
            "complexity_factor": float(out[2]),
        }

    def get_maze_params(self, time_taken: float, mistakes: int,
                        moves: int, optimal_moves: int,
                        current_level: int,
                        prev_params: dict = None,
                        lives_left: int = 3) -> dict:
        """
        Map player performance to concrete maze-generation parameters
        for the specified next level.

        Each dimension adapts independently:
            size      ← time + efficiency performance (fast/efficient → bigger maze)
            dead_ends ← mistakes + lives performance  (accurate/careful → more rocks)
            complexity← overall performance           (all-round good → more complex)

        Parameters
        ----------
        current_level : 2 or 3  (level about to be generated)
        prev_params   : params from previous level (needed for level-3 scaling)
        lives_left    : lives remaining when the level was finished
        """
        factors = self.predict_difficulty(
            time_taken, mistakes, moves, optimal_moves, lives_left)

        sf = factors["size_factor"]         # time + efficiency dimension
        rf = factors["rocks_factor"]        # mistakes + lives dimension
        cf = factors["complexity_factor"]   # overall

        if current_level == 2:
            # Size:       17 (slowest/inefficient player) → 23 (fastest/most efficient)
            raw_size = 17 + sf * 6
            size     = int(raw_size)
            if size % 2 == 0:
                size += 1
            size = max(17, min(23, size))

            # Rocks:      2 (least accurate/most lives lost) → 12 (most accurate/careful)
            dead_ends = max(2, int(2 + rf * 10))

            # Complexity: 0.25 → 0.60
            complexity = round(0.25 + cf * 0.35, 2)

        else:  # level 3 — always harder than level 2
            if prev_params:
                p_size       = prev_params.get("size",       17)
                p_dead_ends  = prev_params.get("dead_ends",   5)
                p_complexity = prev_params.get("complexity", 0.40)
            else:
                p_size, p_dead_ends, p_complexity = 17, 5, 0.40

            # Always at least 2 cells bigger than previous level
            min_size = p_size + 2
            raw_size = min_size + sf * 4
            size     = int(raw_size)
            if size % 2 == 0:
                size += 1
            size = max(min_size, min(27, size))

            # Rocks always increase from level 2
            dead_ends  = max(p_dead_ends + 1, int(p_dead_ends + 2 + rf * 6))

            # Complexity always increases from level 2
            complexity = round(min(0.95, p_complexity + 0.10 + cf * 0.20), 2)

        return {
            "size":             size,
            "dead_ends":        dead_ends,
            "complexity":       complexity,
            "size_factor":       sf,
            "rocks_factor":      rf,
            "complexity_factor": cf,
        }


# Singleton used by app.py
model = AdaptiveMazeModel()
