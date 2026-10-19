"""raw 샌드박스 상태 → 정책 관측, 그리고 행동 → HaxBall 입력 마스크.

이 프로젝트의 유일한 진실 공급원이다. 학습(vecenv)·평가(eval)·녹화(record)·
실전(serve)이 전부 여기를 통과하므로, 에이전트는 네 경로에서 완전히 동일한
특징을 본다.

모든 함수는 환경 N개에 대해 벡터화돼 있다. 핵심은 미러링 — 파랑 팀은 x와 vx의
부호를 뒤집고 빨강/파랑 슬롯을 스왑해서 "나는 +x를 공격한다" 좌표계로 만든다.
덕분에 네트워크 하나가 양쪽 팀을 다 둔다.

기하는 엔진에서 직접 읽었다 (`Utils.getDefaultStadiums()`, Classic):
벽 x=±420 / y=±200, 공은 y=±170에 갇히고, 골대는 x=±370에 포스트 y=±64.
빨강은 +x를, 파랑은 -x를 공격한다.
"""

import numpy as np

# --- Classic stadium constants -------------------------------------------
HALF_W, HALF_H = 420.0, 200.0
GOAL_X, GOAL_POST_Y = 370.0, 64.0
BALL_R, PLAYER_R, KICK_REACH = 10.0, 15.0, 4.0
REACH = PLAYER_R + BALL_R + KICK_REACH          # engine: dist - rb - rp < 4

# --- raw state layout (must match js/envhost.js) --------------------------
BALL, RED, BLUE = 0, 4, 8                        # float offsets, 4 each
CHG_RED, CHG_BLUE, EVENT = 0, 1, 2               # byte offsets
EV_NONE, EV_RED_GOAL, EV_BLUE_GOAL = 0, 1, 2

OBS_DIM = 27
N_ACTIONS = 18                                   # 9 directions x kick on/off

_VSCALE = 5.0
_SCALE = np.array([HALF_W, HALF_H, _VSCALE, _VSCALE], dtype=np.float32)

# hoisted out of build_obs (was rebuilt every call)
_GOAL_THEM = np.array([GOAL_X, 0.0], dtype=np.float32)
_GOAL_MINE = np.array([-GOAL_X, 0.0], dtype=np.float32)
# the x / vx columns of ball, red, blue -- the ones mirrored for blue
_XCOLS = [BALL + 0, BALL + 2, RED + 0, RED + 2, BLUE + 0, BLUE + 2]


def build_obs(state, chg, side, kick_prev):
    """state (N,12) f32, chg (N,2) u8, side (N,) 0=red 1=blue, kick_prev (N,) u8.

    Returns (N, OBS_DIM) f32 in an "I attack +x" frame, so one policy plays
    both colours.
    """
    n = state.shape[0]
    blue = side.astype(bool)
    s = state.astype(np.float32, copy=True)

    # Mirror blue's world so it also attacks +x: flip x and vx (one vectorised
    # write instead of six masked ones), then red and blue slots swap roles.
    s[:, _XCOLS] *= np.where(blue, np.float32(-1.0), np.float32(1.0))[:, None]

    ball = s[:, BALL:BALL + 4]
    me = np.where(blue[:, None], s[:, BLUE:BLUE + 4], s[:, RED:RED + 4])
    opp = np.where(blue[:, None], s[:, RED:RED + 4], s[:, BLUE:BLUE + 4])
    my_chg = np.where(blue, chg[:, CHG_BLUE], chg[:, CHG_RED]).astype(np.float32)
    op_chg = np.where(blue, chg[:, CHG_RED], chg[:, CHG_BLUE]).astype(np.float32)

    d_ball_me = ball[:, :2] - me[:, :2]
    d_ball_opp = ball[:, :2] - opp[:, :2]
    goal_them = _GOAL_THEM - ball[:, :2]
    goal_mine = _GOAL_MINE - ball[:, :2]

    r_me = np.linalg.norm(d_ball_me, axis=1)
    r_opp = np.linalg.norm(d_ball_opp, axis=1)
    r_goal = np.linalg.norm(goal_them, axis=1)

    out = np.empty((n, OBS_DIM), dtype=np.float32)
    out[:, 0:4] = me / _SCALE
    out[:, 4:8] = opp / _SCALE
    out[:, 8:12] = ball / _SCALE
    out[:, 12:14] = d_ball_me / HALF_W
    out[:, 14:16] = d_ball_opp / HALF_W
    out[:, 16:18] = goal_them / HALF_W
    out[:, 18:20] = goal_mine / HALF_W
    out[:, 20] = r_me / HALF_W
    out[:, 21] = r_opp / HALF_W
    out[:, 22] = r_goal / HALF_W
    out[:, 23] = my_chg
    out[:, 24] = op_chg
    out[:, 25] = kick_prev
    out[:, 26] = (r_me < REACH).astype(np.float32)   # ball is kickable now
    return out


# --- actions --------------------------------------------------------------
# action = dir * 2 + kick;  dir = (dx+1)*3 + (dy+1)
UP, DOWN, LEFT, RIGHT, KICK = 1, 2, 4, 8, 16

_DIR = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], dtype=np.int8)
_MASK = np.array(
    [(LEFT if dx < 0 else RIGHT if dx > 0 else 0) | (UP if dy < 0 else DOWN if dy > 0 else 0)
     for dx, dy in _DIR], dtype=np.uint8)
# same table with dx flipped, for blue (whose obs frame is mirrored)
_MASK_MIRROR = np.array(
    [(RIGHT if dx < 0 else LEFT if dx > 0 else 0) | (UP if dy < 0 else DOWN if dy > 0 else 0)
     for dx, dy in _DIR], dtype=np.uint8)


def action_to_input(action, side):
    """action (N,) int, side (N,) 0=red 1=blue -> (N,) uint8 HaxBall input mask."""
    action = np.asarray(action, dtype=np.int64)
    d, k = action >> 1, (action & 1).astype(np.uint8)
    table = np.where(side.astype(bool), _MASK_MIRROR[d], _MASK[d])
    return (table | (k * KICK)).astype(np.uint8)


def kick_bit(action):
    return (np.asarray(action, dtype=np.int64) & 1).astype(np.uint8)
