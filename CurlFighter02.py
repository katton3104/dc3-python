import json
import pathlib

from dc3client import SocketClient
from dc3client.models import Coordinate, Position, State, StoneRotation, Stones, Velocity

from dataclasses import dataclass
import math
from typing import List, Tuple, Optional, Sequence

import numpy as np
from simulator  import StoneSimulator

# 座標位置：バックライン(y=40.234)、ティーライン(y=38.405)、フロントライン(y=36.576)
TEE = Position(x=0.0, y=38.405)
BACK = Position(x=0.0, y=40.234)
FRONT = Position(x=0.0, y=36.576)
HOUSE_RADIUS = 1.83 / 2  # 1.83 m diameter → 0.915 m radius
STONE_RADIUS = 0.145

# Stones から Position を安全に取り出す
def get_stone_position(stones: Stones, team: int, idx: int) -> Optional[Position]:
    coord_list: List[Coordinate] = stones.team0 if team == 0 else stones.team1
    if idx >= len(coord_list):
        return None
    coord = coord_list[idx]
    if coord is None or not coord.position:
        return None
    pos = coord.position[0]
    if pos.x is None or pos.y is None:
        return None
    return pos

# ハウス中心からの距離で全ストーンを昇順ソート
def sort_stones_by_distance(stones: Stones) -> List['StoneRef']:
    refs: List['StoneRef'] = []
    for team in (0, 1):
        for idx in range(8):
            pos = get_stone_position(stones, team, idx)
            dist = math.inf if pos is None else math.hypot(pos.x - TEE.x, pos.y - TEE.y)
            refs.append(StoneRef(team, idx, dist))
    refs.sort(key=lambda r: r.distance)
    return refs

@dataclass
class StoneRef:
    team: int  # 0 or 1
    idx: int   # 0‥7
    distance: float

# ── config.json を読み込み null を置換してファイルへ書き戻し ──
try:
    with open("config.json", "r", encoding="utf-8") as f:
        sim_conf = json.load(f)
    # seed が null の場合は 0 をセット
    for team_key in ("team0", "team1"):
        players = sim_conf.get("game", {}).get("players", {}).get(team_key, [])
        for p in players:
            if p.get("seed") is None:
                p["seed"] = 0
    # 必要なら他の null フィールドも同様に処理
    # 書き戻し
    with open("config.json", "w", encoding="utf-8") as f:
        json.dump(sim_conf, f, indent=4)
except Exception as e:
    print(f"[warn] config.json 読み込み/書き戻しエラー: {e}")

# ── シミュレータ初期化（引数なし） ──
stone_simulator = StoneSimulator()


def estimate_shot_velocity_fcv1(target: Position, target_speed: float, rotation: StoneRotation) -> Tuple[float, float]:
    """
    C++シミュレータで投擲初速 (vx, vy) を計算
    """
    # 角度と目標速度から簡易計算
    angle = math.atan2(target.y - 2.4, target.x)
    vx = math.cos(angle) * target_speed
    vy = math.sin(angle) * target_speed
    return vx, vy


def evaluate_shot(state: State, my_team_idx: int, target: Position, speed: float, rotation: StoneRotation) -> Tuple[float, float, StoneRotation, float]:
    """
    指定したショットをシミュレートし、最終的なストーン配置からスコアを計算する
    """
    vx, vy = estimate_shot_velocity_fcv1(target, speed, rotation)

    # 現在のすべてのストーン位置を NumPy 配列へ
    pos_list = []
    for team in (0, 1):
        for idx in range(8):
            p = get_stone_position(state.stones, team, idx)
            pos_list.append([p.x, p.y] if p else [float('nan'), float('nan')])
    np_positions = np.array(pos_list)

    # シミュレーション実行
    rot_sign = +1 if rotation == StoneRotation.counterclockwise else -1
    shot_vector = np.array([vx, vy, rot_sign])
    np_xv = np.zeros(np_positions.shape[0])
    np_yv = np.zeros(np_positions.shape[0])
    np_av = np.zeros(np_positions.shape[0])

    result, flag = stone_simulator.simulator(
        np_positions, shot_vector, np_xv, np_yv, np_av
    )

    # スコア計算: 自チームのハウス内ストーン数 - 相手チームのハウス内ストーン数
    score = 0.0
    for team_idx in (0, 1):
        for i in range(8):
            x, y = result[team_idx*8 + i]
            if math.hypot(x - TEE.x, y - TEE.y) <= HOUSE_RADIUS:
                score += (1 if team_idx == my_team_idx else -1)
    return vx, vy, rotation, score


class ThinkingAI:
    def decide(self, state: State, my_team: str) -> Tuple[float, float, StoneRotation]:
        """
        複数の候補ショットをシミュレートし、最もスコアが高いショットを選択する
        """
        my_team_idx = 0 if my_team == "team0" else 1
        sorted_refs = sort_stones_by_distance(state.stones)
        in_house_refs = [r for r in sorted_refs if r.distance < (HOUSE_RADIUS + STONE_RADIUS)]

        # 候補ショットを定義
        if not in_house_refs:
            candidates = [
                (TEE, 2.5, StoneRotation.counterclockwise),
                (TEE, 4.0, StoneRotation.counterclockwise),
                (FRONT, 2.5, StoneRotation.counterclockwise),
            ]
        elif sorted_refs[0].team == my_team_idx:
            candidates = [
                (FRONT, 2.5, StoneRotation.counterclockwise),
                (BACK, 2.5, StoneRotation.counterclockwise),
                (BACK, 3.0, StoneRotation.counterclockwise),
            ]
        else:
            candidates = [
                (TEE, 4.0, StoneRotation.counterclockwise),
                (TEE, 4.0, StoneRotation.clockwise),
                (TEE, 3.5, StoneRotation.counterclockwise),
            ]

        # 最適ショット探索
        best = None
        best_score = -math.inf
        for tgt, spd, rot in candidates:
            vx, vy, r, sc = evaluate_shot(state, my_team_idx, tgt, spd, rot)
            if sc > best_score:
                best_score = sc
                best = (vx, vy, r)

        return best


if __name__ == "__main__":
    cli = SocketClient(host="dc3-server", port=10000, client_name="CurlFighter00", auto_start=True, rate_limit=0.1)
    log_dir = pathlib.Path("logs")
    remove_trajectory = True

    my_team = cli.get_my_team()
    cli.logger.info(f"my_team :{my_team}")

    ai = ThinkingAI()

    while True:
        cli.update()

        if (winner := cli.get_winner()) is not None:
            cli.logger.info("WIN" if my_team == winner else "LOSE")
            break

        if cli.get_next_team() != my_team:
            continue

        state = cli.get_match_data().update_list[-1].state
        vx, vy, rot = ai.decide(state, my_team)
        cli.move(x=vx, y=vy, rotation=rot)

    move_info = cli.get_move_info()
    update_list, trajectory_list = cli.get_update_and_trajectory(remove_trajectory)

    update_dict = {}
    for update in update_list:
        update_dict = cli.convert_update(update, remove_trajectory)
    with open("data.json", "w", encoding="UTF-8") as f:
        json.dump(update_dict, f, indent=4)