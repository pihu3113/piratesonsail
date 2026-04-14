# piratesonsail
 A browser-based maze game built with Flask where every player gets a uniquely generated, adaptive maze — powered by a deep learning model that personalizes difficulty in real time.  The core of this project is a feedforward neural network (4 → 16 → 8 → 3) trained using backpropagation and the Adam optimizer.

  Inputs (4 features):
  - Time taken to complete the level
  - Number of mistakes made
  - Path inefficiency (how far from optimal)
  - Lives lost

  Outputs (3 difficulty factors):
  - size_factor — controls maze size
  - rocks_factor — controls number of obstacles
  - complexity_factor — controls maze layout complexity

  The network uses ReLU activations in hidden layers and Sigmoid on the output layer. Weights were trained on 5000 synthetic gameplay sessions using MSE loss with early stopping. A
  retrain.py script allows the model to be fine-tuned on real player data collected automatically during gameplay, making the system progressively smarter over time.

  ---
  Tech Stack

  - Backend: Python, Flask
  - Deep Learning: NumPy (manual backpropagation), PyTorch (training pipeline)
  - Maze Generation: Iterative Depth-First Search (DFS)
  - Pathfinding: Breadth-First Search (BFS) for optimal move calculation
  - Frontend: HTML, CSS, JavaScript
  - Data: Player sessions stored in JSON

  ---
  Key Features

  - Procedurally generated mazes using DFS — unique per player using name + birthday as seed
  - Adaptive difficulty driven by a trained neural network
  - Real player data collection for model retraining
  - Hall of fame tracking all players who complete all 3 levels
  - Pirate-themed story and animations
