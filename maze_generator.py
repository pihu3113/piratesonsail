"""
DFS Maze Generator for Treasure Hunt: The Pirate's Maze
Uses iterative Depth-First Search to generate perfect mazes.
"""
import random
from collections import deque


def generate_maze(size, seed=None):
    """
    Generate a maze using iterative DFS (Depth-First Search).

    Args:
        size (int): Grid size (must be odd, e.g. 15, 17, 19...)
        seed (int): Random seed for reproducibility

    Returns:
        grid (list[list[int]]): 2D grid where 0=wall, 1=path
        start (tuple): (row, col) of start position
        goal (tuple): (row, col) of goal position
    """
    if seed is not None:
        random.seed(seed)

    # Ensure size is odd
    if size % 2 == 0:
        size += 1

    rows = cols = size
    grid = [[0] * cols for _ in range(rows)]

    # Iterative DFS
    start_r, start_c = 1, 1
    grid[start_r][start_c] = 1
    stack = [(start_r, start_c)]

    while stack:
        r, c = stack[-1]
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(directions)

        moved = False
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 < nr < rows - 1 and 0 < nc < cols - 1 and grid[nr][nc] == 0:
                # Carve passage between current cell and neighbor
                grid[r + dr // 2][c + dc // 2] = 1
                grid[nr][nc] = 1
                stack.append((nr, nc))
                moved = True
                break

        if not moved:
            stack.pop()

    # Ensure goal cell is open and reachable
    goal_r, goal_c = rows - 2, cols - 2
    grid[goal_r][goal_c] = 1
    # Carve to goal if isolated (shouldn't happen but safety check)
    if grid[goal_r][goal_c - 1] == 0 and grid[goal_r - 1][goal_c] == 0:
        grid[goal_r][goal_c - 1] = 1

    start = (start_r, start_c)
    goal = (goal_r, goal_c)

    return grid, start, goal


def bfs_path_length(grid, start, goal):
    """
    BFS to find the shortest path length from start to goal.
    Returns path length (number of moves), or -1 if no path.
    """
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False] * cols for _ in range(rows)]
    queue = deque([(start[0], start[1], 0)])
    visited[start[0]][start[1]] = True

    while queue:
        r, c, dist = queue.popleft()
        if (r, c) == goal:
            return dist
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == 1 and not visited[nr][nc]):
                visited[nr][nc] = True
                queue.append((nr, nc, dist + 1))

    return -1


def place_rocks(grid, start, goal, num_rocks, seed=None):
    """
    Place rocks (damage obstacles) in open path cells.
    Rocks are passable but cost the player 1 life.
    Avoids placing rocks at start or goal.

    Returns:
        list of (row, col) tuples for rock positions
    """
    if seed is not None:
        random.seed(seed + 999)

    rows = len(grid)
    cols = len(grid[0])

    open_cells = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if grid[r][c] == 1 and (r, c) != start and (r, c) != goal
    ]

    random.shuffle(open_cells)
    # Place rocks on at most 15% of open cells, up to num_rocks
    max_rocks = min(num_rocks, len(open_cells), int(len(open_cells) * 0.15))
    rocks = open_cells[:max_rocks]
    return rocks


def build_level(size, num_rocks, seed=None):
    """
    Build a complete level: maze + rocks + metadata.
    Guarantees the maze is solvable (BFS verified).

    Returns dict with:
        grid, start, goal, rocks, optimal_moves, rows, cols
    """
    attempt = 0
    while True:
        s = seed + attempt if seed is not None else None
        grid, start, goal = generate_maze(size, seed=s)
        optimal_moves = bfs_path_length(grid, start, goal)

        if optimal_moves > 0:          # solvable — BFS found a path
            break

        attempt += 1                   # extremely rare; try a new seed
        if attempt > 20:
            # Absolute fallback: build the simplest possible corridor maze
            grid = [[0] * size for _ in range(size)]
            for i in range(1, size - 1):
                grid[1][i] = 1
                grid[i][size - 2] = 1
                grid[size - 2][i] = 1
            start = (1, 1)
            goal  = (size - 2, size - 2)
            optimal_moves = bfs_path_length(grid, start, goal)
            break

    rocks = place_rocks(grid, start, goal, num_rocks, seed=seed)
    rows = cols = len(grid)

    return {
        "grid":          grid,
        "start":         list(start),
        "goal":          list(goal),
        "rocks":         [list(r) for r in rocks],
        "optimal_moves": optimal_moves,
        "rows":          rows,
        "cols":          cols,
    }
