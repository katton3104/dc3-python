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

# ショット一覧
# ボタン
DRAW_TEE: Tuple[float, float, StoneRotation] = (0.131725, 2.39969, StoneRotation.counterclockwise)
# ガード
DARW_GUARD:  Tuple[float, float, StoneRotation] = (0.127987, 2.33253, StoneRotation.counterclockwise)
# テイクアウト
TAKEOUT_CENTER: Tuple[float, float, StoneRotation] = (0.06, 3.999955, StoneRotation.counterclockwise)
TAKEOUT_Right_1:  Tuple[float, float, StoneRotation] = (0.08, 3.9992, StoneRotation.counterclockwise)
TAKEOUT_Right_2:  Tuple[float, float, StoneRotation] = (0.10, 3.99875, StoneRotation.counterclockwise)
TAKEOUT_Right_3:  Tuple[float, float, StoneRotation] = (0.12, 3.9982, StoneRotation.counterclockwise)# 赤い円の右端通貨
TAKEOUT_Left_1:  Tuple[float, float, StoneRotation] = (-0.08, 3.9992, StoneRotation.clockwise)
TAKEOUT_Left_2:  Tuple[float, float, StoneRotation] = (-0.10, 3.99875, StoneRotation.clockwise)
TAKEOUT_Left_3:  Tuple[float, float, StoneRotation] = (-0.12, 3.9982, StoneRotation.clockwise)# 赤い円の左端通貨

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

class ThinkingAI:
    def __init__(self):
        # フォーカスゾーンは decide 内で動的に判定するため、初期化不要
        pass

    def decide(self, state: State, my_team: str) -> Tuple[float, float, StoneRotation]:
        # 全ストーンをティー中心から近い順にソート
        sorted_refs = sort_stones_by_distance(state.stones)
        my_team_idx = 0 if my_team == "team0" else 1

        # フォーカスゾーン内のストーンを抽出
        focus_refs: List[StoneRef] = []
        for ref in sorted_refs:
            pos = get_stone_position(state.stones, ref.team, ref.idx)
            if pos is None:
                continue
            # フォーカスゾーンの定義
            if pos.y < (TEE.y + 7 * STONE_RADIUS) and abs(pos.x - TEE.x) <= 7 * STONE_RADIUS:
                focus_refs.append(ref)

        # フォーカスゾーンにストーンがなければボタンショット
        if not focus_refs:
            return DRAW_TEE

        # ナンバーワンストーン（最もティーに近い）を取得
        top = focus_refs[0]
        top_pos = get_stone_position(state.stones, top.team, top.idx)

        # ナンバーワンが自チームならガード
        if top.team == my_team_idx:
            return DARW_GUARD

        # ナンバーワンが相手チームならX座標からテイクアウト種別を選択
        dx = top_pos.x - TEE.x
        adx = abs(dx)
        if adx <= 1 * STONE_RADIUS:
            return TAKEOUT_CENTER
        elif adx <= 3 * STONE_RADIUS:
            return TAKEOUT_Right_1 if dx > 0 else TAKEOUT_Left_1
        elif adx <= 5 * STONE_RADIUS:
            return TAKEOUT_Right_2 if dx > 0 else TAKEOUT_Left_2
        else: # 5*STONE_RADIUS < |dx| ≤ 7*STONE_RADIUS
            return TAKEOUT_Right_3 if dx > 0 else TAKEOUT_Left_3

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