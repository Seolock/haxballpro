"""신경망이 아닌 상대들 — 스크립트 / 랜덤 / 정지, 그리고 얼린 정책 래퍼.

`ScriptedAgent`는 공 뒤로 돌아 들어가 골대 방향으로 정렬되면 킥한다. 커리큘럼
1단계 상대이자 학습 내내 바뀌지 않는 평가 기준선이다 — 리그가 이상한 방향으로
표류하는 걸 막는 앵커 역할을 한다.

전부 `act(obs) → actions` 하나만 구현하므로 환경 입장에서는 정책이든 규칙이든
구분이 없다. 관측은 이미 "나는 +x를 공격한다" 좌표계라서 어느 팀이든 그대로
동작한다.
"""

import numpy as np

from hbrl import obs as O


def _dir_index(dx, dy, dead=0.1):
    sx = np.where(dx > dead, 1, np.where(dx < -dead, -1, 0))
    sy = np.where(dy > dead, 1, np.where(dy < -dead, -1, 0))
    return (sx + 1) * 3 + (sy + 1)


class ScriptedAgent:
    def __init__(self, epsilon=0.05, seed=0):
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        ball_me = obs[:, 12:14] * O.HALF_W                 # ball - me
        goal_ball = obs[:, 16:18] * O.HALF_W               # opponent goal - ball
        in_reach = obs[:, 26] > 0

        gdir = goal_ball / (np.linalg.norm(goal_ball, axis=1, keepdims=True) + 1e-6)
        # stand this far behind the ball, on the far side from the goal
        approach = ball_me - gdir * (O.PLAYER_R + O.BALL_R)

        # aligned == I am behind the ball relative to the goal
        bdir = ball_me / (np.linalg.norm(ball_me, axis=1, keepdims=True) + 1e-6)
        aligned = (bdir * gdir).sum(axis=1) > 0.8

        target = np.where(aligned[:, None], ball_me, approach)
        norm = np.linalg.norm(target, axis=1, keepdims=True) + 1e-6
        d = target / norm

        act = _dir_index(d[:, 0], d[:, 1]) * 2 + (in_reach & aligned).astype(np.int64)

        n = obs.shape[0]
        flip = self.rng.random(n) < self.epsilon
        act[flip] = self.rng.integers(0, O.N_ACTIONS, flip.sum())
        return act


class GoalieAgent:
    """Parks on the line between the ball and its own goal, clears on contact.

    The situation the self-play league never produced: the 1.02B-step policy
    scored on this agent in 14 of 256 episodes while beating `ScriptedAgent`
    161-0. `ScriptedAgent` alone is a useless anchor because pure dribbling
    beats it -- this one has to be shot around.
    """

    def __init__(self, depth=90.0):
        self.depth = depth

    def act(self, obs):
        scale = np.array([O.HALF_W, O.HALF_H], np.float32)
        ball = obs[:, 8:10] * scale
        me = obs[:, 0:2] * scale
        own = np.array([-O.GOAL_X, 0.0], np.float32)       # I attack +x, so mine is -x

        out = ball - own
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-6
        in_reach = obs[:, 26] > 0
        # stand on the ball-goal line; once the ball is on me, drive it away
        target = np.where(in_reach[:, None], out, own + out * self.depth - me)
        d = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-6)
        return _dir_index(d[:, 0], d[:, 1]) * 2 + in_reach.astype(np.int64)


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs):
        return self.rng.integers(0, O.N_ACTIONS, obs.shape[0])


class PolicyAgent:
    """A frozen SB3 policy used as a league opponent."""

    def __init__(self, policy):
        self.policy = policy

    def act(self, obs):
        a, _ = self.policy.predict(obs, deterministic=False)
        return np.asarray(a).reshape(-1)


class IdleAgent:
    """Stands still. Useful as a fixed reference and in tests."""

    def act(self, obs):
        return np.full(obs.shape[0], 4 * 2, dtype=np.int64)      # dir (0,0), no kick


def load_agent(spec):
    """스펙 문자열 -> 에이전트. .zip 체크포인트 경로이거나
    `baseline` / `goalie` / `random` / `idle` 중 하나."""
    if spec == "baseline":
        return ScriptedAgent(epsilon=0.0)
    if spec == "goalie":
        return GoalieAgent()
    if spec == "random":
        return RandomAgent()
    if spec == "idle":
        return IdleAgent()
    from stable_baselines3 import PPO          # 무거우니 필요할 때만 가져온다
    policy = PPO.load(spec, device="cpu").policy
    policy.set_training_mode(False)
    return PolicyAgent(policy)
