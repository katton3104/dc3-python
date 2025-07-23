import json
import pathlib

from dc3client import SocketClient
from dc3client.models import Coordinate, Position, State, StoneRotation, Stones, Velocity

from dataclasses import dataclass
import math
from typing import List, Tuple, Optional, Sequence
import simulator

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
def sort_stones_by_distance(stones: Stones) -> List[StoneRef]:
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

def _poly(a: Sequence[float], x: float) -> float:
    return sum(coef * x ** i for i, coef in enumerate(reversed(a)))

def estimate_shot_velocity_fcv1(target: Position, target_speed: float, rotation: StoneRotation) -> tuple[float, float]:
    # 入力チェック
    assert 0.0 <= target_speed <= 4.0, "target_speed must be in [0, 4]"

    # StoneRotation enum からシミュレータ向けの回転符号を生成
    # ここでは、ccw → +1, cw → -1 とする
    # 入力チェック
    if not (0.0 <= target_speed <= 4.0):
        raise ValueError(f"target_speed must be in [0, 4], got {target_speed}")

    # simulator モジュールの関数を呼び出し
    # 関数名・引数順はドキュメントに合わせて適宜読み替えてください
    vx, vy = simulator.estimate_shot_velocity(
        float(target.x),
        float(target.y),
        float(target_speed),
        rot_sign
    )

    return vx, vy

class ThinkingAI:
    def decide(self, state: State, my_team: str) -> Tuple[float, float, StoneRotation]:
        """
        ・ハウス内にストーンがなければハウス中心へのドロー
        ・それ以外はナンバーワンストーンが自チームならガード、相手チームならテイクアウト
        """
        sorted_refs = sort_stones_by_distance(state.stones)
        my_team_idx = 0 if my_team == "team0" else 1

        # 1) ハウス内のストーン抽出
        stones_in_house = [
            ref for ref in sorted_refs
            if ref.distance < (HOUSE_RADIUS + STONE_RADIUS)
        ]
        if not stones_in_house:
            # ハウス内ストーンゼロ → ハウス中心へ置く
            return CENTER_SHOT

        # 2) ハウス内にストーンあり → 最も近いストーンで判断
        top = sorted_refs[0]
        if top.team == my_team_idx:
            # 自チームリード → ガード
            return GUARD_SHOT
        else:
            # 相手リード → テイクアウト（ハウス中心）
            return CENTER_SHOT

if __name__ == "__main__":
    cli = SocketClient(host="dc3-server", port=10001, client_name="CurlFighter01", auto_start=True, rate_limit=0.1)
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