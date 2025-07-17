import json
import pathlib

from dc3client import SocketClient
from dc3client.models import Coordinate, Position, State, StoneRotation, Stones

import math
from dataclasses import dataclass

from typing import List, Tuple, Optional, Sequence
# 座標位置：バックライン(y=40.234)、ティーライン(y=38.405)、フロントライン(y=36.576)
TEE = Position(x=0.0, y=38.405)
BACK = Position(x=0.0, y=40.234)
FRONT = Position(x=0.0, y=36.576)
HOUSE_RADIUS = 1.83 / 2  # 1.83 m diameter → 0.915 m radius
STONE_RADIUS = 0.145

# ボタン
CENTER_SHOT: Tuple[float, float, StoneRotation] = (0.131725, 2.39969, StoneRotation.counterclockwise)
# ガード
GUARD_SHOT:  Tuple[float, float, StoneRotation] = (0.127987, 2.33253, StoneRotation.counterclockwise)

@dataclass
class StoneRef:
    team: int # 0 or 1
    idx: int # 0‥7
    distance: float

# Stones から Position を安全に取り出す
def get_stone_position(stones: Stones, team: int, idx: int) -> Optional[Position]:
    """Return current Position of the stone or *None* if out of play."""
    coord_list: List[Coordinate] = stones.team0 if team == 0 else stones.team1
    if idx >= len(coord_list):
        return None
    coord = coord_list[idx]
    if coord is None or not coord.position:
        return None
    # position[0] が現時点の座標（仕様より）
    pos = coord.position[0]
    if pos.x is None or pos.y is None:
        return None
    return pos

# ハウス中心からの距離で全ストーンを昇順ソート
def sort_stones_by_distance(stones: Stones) -> List[StoneRef]:
    """。"""
    refs: List[StoneRef] = []
    for team in (0, 1):
        for idx in range(8):
            pos = get_stone_position(stones, team, idx)
            if pos is None:
                dist = math.inf
            else:
                dist = math.hypot(pos.x - TEE.x, pos.y - TEE.y)
            refs.append(StoneRef(team, idx, dist))
    refs.sort(key=lambda r: r.distance)
    return refs

# estimate_shot_velocity_fcv1で使用
def _poly(a: Sequence[float], x: float) -> float:
    return sum(coef * x ** i for i, coef in enumerate(reversed(a)))

# 単回帰近似式と delta_angle 簡易近似をそのまま使用
def estimate_shot_velocity_fcv1(target: Position, target_speed: float, rotation: StoneRotation) -> Tuple[float, float]:
    
    assert 0.0 <= target_speed <= 4.0
    target_r = math.hypot(target.x, target.y)
    assert target_r > 0.0

    if target_speed <= 0.05:
        kC0 = (0.0005048122574925176, 0.2756242531609261)
        kC1 = (0.00046669575066030805, -29.898958358378636, -0.0014030973174948508)
        kC2 = (0.13968687866736632, 0.41120940058777616)
    elif target_speed <= 1.0:
        kC0 = (-0.0014309170115803444, 0.9858457898438147)
        kC1 = (-0.0008339331735471273, -29.86751291726946, -0.19811799977982522)
        kC2 = (0.13967323742978, 0.42816312110477517)
    else:
        kC0 = (1.0833113118071224e-06, -0.00012132851917870833, 0.004578093297561233, 0.9767006869364527)
        kC1 = (0.07950648211492622, -8.228225657195706, -0.05601306077702578)
        kC2 = (0.14140440186382008, 0.3875782508767419)

    def c0(r: float) -> float:
        return _poly(kC0, r)

    def c1(r: float) -> float:
        return -kC1[0] * math.log(r + kC1[1]) + kC1[2]

    def c2(r: float) -> float:
        return kC2[0] * r + kC2[1]

    v0_mag = math.sqrt(c0(target_r) * target_speed ** 2 + c1(target_r) * target_speed + c2(target_r))
    assert target_speed < v0_mag

    # --- 2. get delta angle by one‑shot simulation (simplified) -----------
    # NOTE: A full physical simulation is OUT OF SCOPE for this minimal port.
    # We approximate the angular correction term (delta_angle) by the linear
    # factor used in the original code when no sim is available.
    rotation_factor = 1.0 if rotation == StoneRotation.counterclockwise else -1.0

    # Original C++ sim computes delta = stone_pos_when_speed_le_target - target
    # Here we approximate with empirical constant since full sim unavailable.
    # **You SHOULD replace this with proper simulator calls** if accuracy is
    # required.
    delta_const = 0.25  # 簡易近似
    delta_angle = rotation_factor * delta_const / target_r

    target_angle = math.atan2(target.y, target.x)
    v0_angle = target_angle + delta_angle

    vx = v0_mag * math.cos(v0_angle)
    vy = v0_mag * math.sin(v0_angle)
    return vx, vy

class ThinkingAI:
    """Very small rule‑based AI using the fixed helpers."""

    def decide(self, state: State, my_team: str) -> Tuple[float, float, StoneRotation]:
        # ハウス内の最も中心に近いストーンを取得
        sorted_refs = sort_stones_by_distance(state.stones)
        top = sorted_refs[0]

        # 自分のチーム番号 (0/1) を算出
        my_team_idx = 0 if my_team == "team0" else 1

        in_house = top.distance < (HOUSE_RADIUS + STONE_RADIUS)
        opponent_leading = in_house and top.team != my_team_idx

        if opponent_leading:
            # テイクアウト狙い
            target_pos = get_stone_position(state.stones, top.team, top.idx)
            if target_pos is not None:
                vx, vy = estimate_shot_velocity_fcv1(target_pos, 3.0, StoneRotation.clockwise)
                return vx, vy, StoneRotation.clockwise

        # それ以外は交互にセンター／ガード
        if state.shot % 2 == 0:
            return CENTER_SHOT
        else:
            return GUARD_SHOT


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
