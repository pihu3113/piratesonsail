"""
Treasure Hunt: The Pirate's Maze — Flask Backend
"""

import json
import os
import time as _time

from flask import (Flask, jsonify, redirect, render_template,
                   request, session, url_for)

from maze_generator import build_level
from model import model as dl_model

app = Flask(__name__)
app.secret_key = "pirates_secret_key_2024_arrr"

PLAYERS_FILE       = os.path.join(os.path.dirname(__file__), "players.json")
GAMEPLAY_DATA_FILE = os.path.join(os.path.dirname(__file__), "gameplay_data.json")



def _load_players():
    try:
        with open(PLAYERS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save_players(players):
    with open(PLAYERS_FILE, "w") as f:
        json.dump(players, f, indent=2)


def _record_winner():
    """
    Save the current session's player to the hall of fame.
    Safe to call multiple times — will not create duplicate entries
    within the same session (tracked by session["saved"]).
    """
    if session.get("saved"):
        return                          # already saved this run

    name  = session.get("player_name", "Unknown Captain")
    stats = session.get("stats", [])

    players = _load_players()
    players.append({
        "name":      name,
        "birthday":  session.get("birthday", ""),
        "completed": True,
        "stats":     stats,
        "date":      _time.strftime("%Y-%m-%d"),
    })
    _save_players(players)
    session["saved"] = True             # mark so we don't double-save


def _save_gameplay_sample(input_stats: dict, next_stats: dict):
    """
    Save one real training sample to gameplay_data.json.

    input_stats  — stats from level N (these were fed into the DL model)
    next_stats   — stats from level N+1 (how hard/easy that level actually felt)

    This gives us delayed feedback: we know what the model predicted for level N+1,
    and now we see how the player actually performed — so we can compute better labels.
    """
    sample = {
        "input": {
            "time_norm":       round(min(input_stats["time"] / 300.0, 1.0), 4),
            "mistake_norm":    round(min(input_stats["mistakes"] / 15.0, 1.0), 4),
            "inefficiency":    round(1.0 - input_stats["efficiency"], 4),
            "lives_lost_norm": round((3 - input_stats["lives_left"]) / 3.0, 4),
        },
        "next_level_performance": {
            "time_norm":       round(min(next_stats["time"] / 300.0, 1.0), 4),
            "mistake_norm":    round(min(next_stats["mistakes"] / 15.0, 1.0), 4),
            "inefficiency":    round(1.0 - next_stats["efficiency"], 4),
            "lives_lost_norm": round((3 - next_stats["lives_left"]) / 3.0, 4),
        },
    }
    try:
        with open(GAMEPLAY_DATA_FILE) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []
    data.append(sample)
    with open(GAMEPLAY_DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Fixed Level-1 params ──────────────────────────────────────────────────────
LEVEL1 = {"size": 15}


def _player_seed(level: int = 1) -> int:
    """
    Derive a deterministic seed that is unique to this player + level.
    Using name + birthday gives each player a different maze layout
    while remaining reproducible (same player always gets the same maze).
    """
    name     = session.get("player_name", "")
    birthday = session.get("birthday", "")
    raw      = hash(f"{name}:{birthday}:lvl{level}")
    return abs(raw) % (2 ** 31)

# ── Page Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def login():
    return render_template("login.html")


@app.route("/story", methods=["POST"])
def story():
    name     = request.form.get("name", "Captain").strip() or "Captain"
    birthday = request.form.get("birthday", "")
    session.clear()
    session["player_name"]      = name
    session["birthday"]         = birthday
    session["level"]            = 1          # current level to play
    session["completed_level"]  = 0          # last level just finished
    session["lives"]            = 3
    session["stats"]            = []
    session["level2_params"]    = None
    session["level3_params"]    = None
    session["saved"]            = False
    return redirect(url_for("game"))


@app.route("/game")
def game():
    if "player_name" not in session:
        return redirect(url_for("login"))
    level = session.get("level", 1)
    name  = session.get("player_name", "Captain")
    lives = session.get("lives", 3)
    return render_template("game.html", level=level, name=name, lives=lives)


@app.route("/level_complete")
def level_complete():
    if "player_name" not in session:
        return redirect(url_for("login"))

    # Use the level that was JUST finished, not the next one to play
    completed = session.get("completed_level", 1)
    name      = session.get("player_name", "Captain")
    stats     = session.get("stats", [])
    last      = stats[-1] if stats else {}
    next_lvl  = session.get("level", completed + 1)

    return render_template("level_complete.html",
                           level=completed,
                           next_level=next_lvl,
                           name=name,
                           stats=last)


@app.route("/win")
def win():
    if "player_name" not in session:
        return redirect(url_for("login"))

    # Guarantee the player is saved even if the AJAX call somehow failed
    if session.get("level", 1) > 3 or session.get("completed_level", 0) >= 3:
        _record_winner()

    name  = session.get("player_name", "Captain")
    stats = session.get("stats", [])
    return render_template("win.html", name=name, stats=stats)


@app.route("/history")
def history():
    players = _load_players()
    return render_template("history.html", players=players)


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/get_maze", methods=["POST"])
def api_get_maze():
    """Return maze data for the current level."""
    if "player_name" not in session:
        return jsonify({"error": "not logged in"}), 401

    level = session.get("level", 1)

    if level == 1:
        # Level 1 is player-unique but fixed difficulty (baseline)
        data = build_level(LEVEL1["size"], num_rocks=0, seed=_player_seed(1))
    elif level == 2:
        params    = session.get("level2_params") or {"size": 17, "dead_ends": 3, "complexity": 0.30}
        num_rocks = params.get("dead_ends", 3)
        data      = build_level(params["size"], num_rocks=num_rocks, seed=_player_seed(2))
    else:   # level 3
        params    = session.get("level3_params") or {"size": 19, "dead_ends": 5, "complexity": 0.40}
        num_rocks = params.get("dead_ends", 5)
        data      = build_level(params["size"], num_rocks=num_rocks, seed=_player_seed(3))

    data["level"] = level
    data["lives"] = session.get("lives", 3)
    return jsonify(data)


@app.route("/api/level_complete", methods=["POST"])
def api_level_complete():
    """
    Called by the browser when a level is finished.
    Records stats, runs DL model for next level, saves winner if done.
    """
    if "player_name" not in session:
        return jsonify({"error": "not logged in"}), 401

    body          = request.get_json(force=True)
    level         = session.get("level", 1)   # level just finished
    time_taken    = float(body.get("time_taken", 60))
    mistakes      = int(body.get("mistakes", 0))
    moves         = int(body.get("moves", 1))
    optimal_moves = int(body.get("optimal_moves", 1))
    lives_left    = int(body.get("lives", 3))

    efficiency = round(optimal_moves / max(moves, 1), 3)

    stat_entry = {
        "level":         level,
        "time":          round(time_taken, 1),
        "mistakes":      mistakes,
        "moves":         moves,
        "optimal_moves": optimal_moves,
        "efficiency":    efficiency,
        "lives_left":    lives_left,
    }

    stats = session.get("stats", [])
    stats.append(stat_entry)
    session["stats"]           = stats
    session["completed_level"] = level          # remember which level just finished
    session["lives"]           = 3              # reset lives for next level

    # ── Real data collection: save input→feedback pair ────────────────
    # When level 2 finishes: level 1 stats were input, level 2 is feedback
    # When level 3 finishes: level 2 stats were input, level 3 is feedback
    if level in (2, 3) and len(stats) >= 2:
        _save_gameplay_sample(input_stats=stats[-2], next_stats=stats[-1])

    # ── DL model: prepare next level parameters ───────────────────────
    next_level = level + 1

    if next_level == 2:
        params = dl_model.get_maze_params(
            time_taken, mistakes, moves, optimal_moves,
            current_level=2, prev_params=None, lives_left=lives_left)
        session["level2_params"] = params

    elif next_level == 3:
        prev   = session.get("level2_params")
        params = dl_model.get_maze_params(
            time_taken, mistakes, moves, optimal_moves,
            current_level=3, prev_params=prev, lives_left=lives_left)
        session["level3_params"] = params
    else:
        params = {}

    session["level"] = next_level               # advance to next level

    # ── Save to hall of fame when all 3 levels are done ───────────────
    if next_level > 3:
        _record_winner()

    return jsonify({
        "success":    True,
        "next_level": next_level,
        "params":     params,
        "efficiency": efficiency,
    })


@app.route("/api/update_lives", methods=["POST"])
def api_update_lives():
    """Sync lives from frontend to session."""
    if "player_name" not in session:
        return jsonify({"error": "not logged in"}), 401
    body  = request.get_json(force=True)
    lives = int(body.get("lives", 3))
    session["lives"] = lives
    return jsonify({"lives": lives})


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
