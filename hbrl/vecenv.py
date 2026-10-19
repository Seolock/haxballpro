"""node-haxball 샌드박스 룸 위에 올린 SB3 VecEnv.

구성: node 프로세스 W개 × 룸 K개. node는 물리와 시나리오 초기 배치만 담당하고,
이 모듈이 관측·보상·에피소드 경계·커리큘럼·자기대전 리그를 전부 소유한다.

에피소드는 `hbrl/scenarios.py`의 시나리오 하나다. 커리큘럼 모드에서는 성공률이
임계치를 넘을 때마다 다음 시나리오가 해금되고, 전부 풀면 `full_game`으로
넘어간다 — 그때부터는 골이 나도 에피소드가 끝나지 않고 엔진이 상대 킥오프로
이어가므로 긴 공방과 실점 직후 수비가 비로소 데이터에 들어온다.

보상은 골 ±10과 시간 패널티뿐이다. 보조(dense) 보상은 없다 — 부트스트랩은
시나리오 0(빈 골대 앞)이 담당한다.

관측은 `obs.py`가 "나는 +x를 공격한다" 좌표계로 미러링하므로 네트워크 하나가
양쪽 팀을 다 학습한다.
"""

import collections
import os
import subprocess

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from hbrl import obs as O
from hbrl import scenarios as S

JS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "js")

FLAG_STEP, FLAG_RESET, FLAG_READ, FLAG_SCENARIO = 0, 1, 2, 3
REC_F32, REC_U8 = 12, 3

MAX_POOL = 12               # league snapshots kept alongside the scripted anchors

GOAL_REWARD = 10.0
# A goalless episode has to hurt. With draws free, both sides converge on
# stalling -- attacking risks a counter for no expected gain. Real 1v1 is played
# to a score limit, where running the clock down wins nothing.
TIME_PENALTY = 2.0 / 600

# per-scenario lookup tables, indexed by scenario id
_LIMIT = np.array([s.limit for s in S.SCENARIOS], dtype=np.int32)
_DEFEND = np.array([s.defend for s in S.SCENARIOS], dtype=bool)
_CLEAR = np.array([s.clear for s in S.SCENARIOS], dtype=bool)
_KICKOFF = np.array([s.kickoff for s in S.SCENARIOS], dtype=bool)
_CONT = np.array([s.continuous for s in S.SCENARIOS], dtype=bool)
_OPP = [s.opponent for s in S.SCENARIOS]


class _Hosts:
    """W node processes speaking the fixed-size binary protocol of envhost.js."""

    def __init__(self, workers, rooms, seed, skip=1):
        self.workers, self.rooms = workers, rooms
        self.n = workers * rooms
        self.req_size = rooms * 3
        self.res_size = rooms * (REC_F32 * 4 + REC_U8)
        self.procs = [
            subprocess.Popen(
                ["node", "envhost.js", str(rooms), str(seed + w), str(skip)],
                cwd=JS_DIR, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            )
            for w in range(workers)
        ]
        self.state = np.zeros((self.n, REC_F32), dtype=np.float32)
        self.bytes = np.zeros((self.n, REC_U8), dtype=np.uint8)

    def exchange(self, req):
        """req (n,3) uint8 -> fills self.state / self.bytes."""
        req = np.ascontiguousarray(req, dtype=np.uint8)
        for w, p in enumerate(self.procs):
            p.stdin.write(req[w * self.rooms:(w + 1) * self.rooms].tobytes())
            p.stdin.flush()
        for w, p in enumerate(self.procs):
            buf = p.stdout.read(self.res_size)
            if len(buf) != self.res_size:
                raise RuntimeError(f"envhost {w} died (got {len(buf)}/{self.res_size} bytes)")
            sl = slice(w * self.rooms, (w + 1) * self.rooms)
            self.state[sl] = np.frombuffer(buf, np.float32, self.rooms * REC_F32).reshape(-1, REC_F32)
            self.bytes[sl] = np.frombuffer(buf, np.uint8, self.rooms * REC_U8,
                                           self.rooms * REC_F32 * 4).reshape(-1, REC_U8)

    def close(self):
        for p in self.procs:
            try:
                p.stdin.close()
                p.wait(timeout=5)
            except Exception:
                p.kill()


class HaxballVecEnv(VecEnv):
    """One scenario per episode; `curriculum=True` walks through them in order."""

    render_mode = None

    def __init__(self, workers=4, rooms=64, seed=0, frame_skip=2,
                 scenario=6, curriculum=False):
        self.hosts = _Hosts(workers, rooms, seed, skip=frame_skip)
        n = self.hosts.n
        super().__init__(n, spaces.Box(-np.inf, np.inf, (O.OBS_DIM,), np.float32),
                         spaces.Discrete(O.N_ACTIONS))
        self.rng = np.random.default_rng(seed)
        self.frame_skip = frame_skip
        self.curriculum = curriculum
        self.pinned = scenario                  # used when curriculum is off

        self.side = self.rng.integers(0, 2, n).astype(np.int8)     # 0 red, 1 blue
        self.scn = np.full(n, 0 if curriculum else scenario, dtype=np.int32)
        self.opp_idx = np.zeros(n, dtype=np.int32)
        self.ticks = np.zeros(n, dtype=np.int32)
        self.kick_prev = np.zeros(n, dtype=np.float32)
        self.opp_kick_prev = np.zeros(n, dtype=np.float32)
        self.ep_reward = np.zeros(n, dtype=np.float32)
        self.point_reward = np.zeros(n, dtype=np.float32)          # since the last goal
        self.point_ticks = np.zeros(n, dtype=np.int32)

        # League. The scripted agents sit at the front; the trainer appends policy
        # snapshots behind them and registers the live policy as `latest`.
        # `IdleAgent` is last and is only ever handed out by a scenario that asks
        # for it -- as a league opponent it would stand on its own kick-off and
        # fence the learner out of the circle for the whole episode.
        from hbrl.baseline import GoalieAgent, IdleAgent, RandomAgent, ScriptedAgent
        self.opponents = [ScriptedAgent(epsilon=0.05, seed=seed),
                          ScriptedAgent(epsilon=0.4, seed=seed + 7),
                          RandomAgent(seed=seed + 1),
                          GoalieAgent()]                 # depth 90: a sweeper
        self.n_league = len(self.opponents)      # scripted agents a league may draw
        # A depth-90 goalie stands on the ball in `shoot_with_keeper` and nobody
        # ever scores (measured 7% for random and scripted alike). The scenario
        # keeper hugs its line instead, which leaves a real angle to beat.
        self.opponents += [GoalieAgent(depth=45), IdleAgent()]
        self.idx_goalie, self.idx_keeper, self.idx_idle = 3, 4, 5
        self.n_fixed = len(self.opponents)
        self.latest = None                       # set via set_learner()
        self.league = False                      # curriculum: scripted mix at first

        # curriculum state
        self.unlocked = 1 if curriculum else len(S.CURRICULUM)
        self.history = [collections.deque(maxlen=S.WINDOW) for _ in S.SCENARIOS]
        self.steps_here = 0                      # env steps spent on the newest one

        self._actions = None
        self._restart(np.ones(n, bool), first=True)

    # --- plumbing ---------------------------------------------------------
    def _req(self, flag, learner_mask=None, opp_mask=None):
        req = np.zeros((self.num_envs, 3), dtype=np.uint8)
        req[:, 0] = flag
        if learner_mask is not None:
            red_is_learner = self.side == 0
            req[:, 1] = np.where(red_is_learner, learner_mask, opp_mask)
            req[:, 2] = np.where(red_is_learner, opp_mask, learner_mask)
        return req

    def _make_obs(self):
        return O.build_obs(self.hosts.state, self.hosts.bytes, self.side, self.kick_prev)

    def _opp_obs(self):
        return O.build_obs(self.hosts.state, self.hosts.bytes,
                           1 - self.side, self.opp_kick_prev)

    def set_learner(self, model):
        """Register the live policy; it becomes the `latest` league opponent."""
        self.latest = model

    def add_snapshot(self, agent):
        """Add a frozen policy to the league and keep the pool bounded.

        Thinning rewrites the pool, so any env already pointing into it holds a
        stale index -- one that now names a different agent, or none at all. An
        index past the end matches no branch in `_opp_actions` and the opponent
        silently plays action 0 (run down-left, never kick) for the rest of its
        episode, handing the learner free wins. The pool and `opp_idx` are one
        invariant, so they get one owner: redraw the affected envs right here.
        """
        self.opponents.append(agent)
        nf = self.n_fixed
        if len(self.opponents) > MAX_POOL + nf:
            # keep the newest, thin the tail, so old styles survive
            self.opponents = self.opponents[:nf] + self.opponents[nf::2][-MAX_POOL:]
        stale = self.opp_idx >= nf
        if stale.any():
            self._assign_opponents(stale, self.scn)

    def _opp_actions(self, obs):
        act = np.zeros(self.num_envs, dtype=np.int64)
        for i, agent in enumerate(self.opponents):
            m = self.opp_idx == i
            if m.any():
                act[m] = agent.act(obs[m])
        m = self.opp_idx == -1                                   # -1 == latest
        if m.any():
            if self.latest is None:
                act[m] = self.rng.integers(0, O.N_ACTIONS, m.sum())
            else:
                act[m], _ = self.latest.predict(obs[m], deterministic=False)
        return act

    # --- curriculum -------------------------------------------------------
    def success_rate(self, sid):
        h = self.history[sid]
        return float(np.mean(h)) if len(h) == h.maxlen else float("nan")

    def _maybe_unlock(self):
        if not self.curriculum or self.unlocked >= len(S.CURRICULUM):
            return
        r = self.success_rate(self.unlocked - 1)
        spent = self.steps_here >= S.UNLOCK_BUDGET
        if (r == r and r >= S.UNLOCK_RATE) or spent:             # r == r: not NaN
            self.unlocked += 1
            self.steps_here = 0
            why = "budget spent" if spent else f"success {r:.2f}"
            print(f"[curriculum] unlocked {self.unlocked - 1} "
                  f"({S.SCENARIOS[self.unlocked - 1].name}); {why}")

    @property
    def in_full_game(self):
        return self.curriculum and self.unlocked >= len(S.CURRICULUM)

    def _pick_scenarios(self, k):
        """Newest unlocked scenario 60% of the time, the rest spread over the
        ones already cleared -- dropping them entirely just makes it forget."""
        if not self.curriculum:
            return np.full(k, self.pinned, dtype=np.int32)
        if self.in_full_game:
            out = np.full(k, S.FULL_GAME.id, dtype=np.int32)
            drill = self.rng.random(k) < 0.1                     # anchor + diagnostic
            out[drill] = self.rng.integers(0, len(S.CURRICULUM), drill.sum())
            return out
        newest = self.unlocked - 1
        out = np.full(k, newest, dtype=np.int32)
        older = self.rng.random(k) >= 0.6
        if newest > 0:
            out[older] = self.rng.integers(0, self.unlocked, older.sum())
        return out

    def _assign_opponents(self, mask, scn):
        """Scenario-forced opponents win; the rest draw from the league."""
        for name, idx in (("goalie", self.idx_goalie), ("keeper", self.idx_keeper),
                          ("idle", self.idx_idle), ("scripted", 0)):
            m = mask & (np.array(_OPP, dtype=object)[scn] == name)
            if m.any():
                self.opp_idx[m] = idx
        league = mask & (np.array(_OPP, dtype=object)[scn] == "league")
        k = int(league.sum())
        if k == 0:
            return
        if not self.league:
            self.opp_idx[league] = self.rng.integers(0, self.n_league, k)
            return
        # 40% latest, 50% snapshot pool, 10% scripted anchor
        r = self.rng.random(k)
        pick = np.full(k, -1, dtype=np.int32)
        pool = np.arange(self.n_fixed, len(self.opponents))
        if len(pool):
            sel = r < 0.6
            pick[sel] = self.rng.choice(pool, sel.sum())
        anchor = r < 0.1
        pick[anchor] = self.rng.integers(0, self.n_league, anchor.sum())
        self.opp_idx[league] = pick

    # --- episode restart --------------------------------------------------
    def _restart(self, mask, first=False):
        """Give every env in `mask` a fresh scenario; leave the others untouched."""
        k = int(mask.sum())
        if k == 0:
            return
        scn = self.scn.copy()
        scn[mask] = self._pick_scenarios(k)
        mirror = self.rng.integers(0, 2, self.num_envs).astype(np.uint8)

        req = np.zeros((self.num_envs, 3), dtype=np.uint8)
        req[:, 0] = FLAG_READ
        kick = mask & _KICKOFF[scn]
        lay = mask & ~_KICKOFF[scn]
        req[kick, 0] = FLAG_RESET                     # node picks the kick-off side
        req[lay, 0] = FLAG_SCENARIO
        req[lay, 1] = scn[lay]
        req[lay, 2] = mirror[lay]
        self.hosts.exchange(req)

        self.scn = scn
        # a laid-out scenario is written in red's frame, so `mirror` decides the
        # learner's colour; a kick-off is symmetric and just gets a coin flip
        self.side[lay] = mirror[lay].astype(np.int8)
        self.side[kick] = self.rng.integers(0, 2, int(kick.sum())).astype(np.int8)
        self.ticks[mask] = 0
        self.point_ticks[mask] = 0
        self.kick_prev[mask] = 0
        self.opp_kick_prev[mask] = 0
        self.ep_reward[mask] = 0
        self.point_reward[mask] = 0
        self._assign_opponents(mask, scn)
        self._obs = self._make_obs()

    # --- VecEnv API -------------------------------------------------------
    def reset(self):
        self._restart(np.ones(self.num_envs, bool))
        return self._obs

    def step_async(self, actions):
        self._actions = np.asarray(actions).reshape(-1)

    def step_wait(self):
        act = self._actions
        opp_act = self._opp_actions(self._opp_obs())
        learner_mask = O.action_to_input(act, self.side)
        opp_mask = O.action_to_input(opp_act, 1 - self.side)

        # envhost.js runs the frame_skip internally: one round-trip, not
        # `frame_skip` of them. Inputs are held for the whole window either way.
        self.hosts.exchange(self._req(FLAG_STEP, learner_mask, opp_mask))
        event = self.hosts.bytes[:, O.EVENT].copy()

        self.kick_prev = O.kick_bit(act).astype(np.float32)
        self.opp_kick_prev = O.kick_bit(opp_act).astype(np.float32)
        self.ticks += 1
        self.point_ticks += 1
        self.steps_here += self.num_envs
        self._obs = self._make_obs()

        # a goal counts for the learner iff the scoring team is its colour
        scored = ((event == O.EV_RED_GOAL) & (self.side == 0)) | \
                 ((event == O.EV_BLUE_GOAL) & (self.side == 1))
        conceded = (event != O.EV_NONE) & ~scored

        rew = (np.where(scored, GOAL_REWARD, 0.0)
               - np.where(conceded, GOAL_REWARD, 0.0)).astype(np.float32) - TIME_PENALTY
        self.ep_reward += rew
        self.point_reward += rew

        cont = _CONT[self.scn]
        defend = _DEFEND[self.scn]
        cleared = _CLEAR[self.scn] & (self._obs[:, 8] > 0)   # ball past halfway
        timeout = self.ticks >= _LIMIT[self.scn]

        # full_game keeps running through goals -- the engine takes it to the
        # conceding side's kick-off, which is where defence after a goal comes from
        done = np.where(cont, timeout,
                        scored | conceded | timeout | cleared)
        # a clearance scenario is only passed by actually getting the ball out --
        # merely surviving the clock is what made a random policy score 70%
        success = np.where(_CLEAR[self.scn], cleared,
                           np.where(defend, ~conceded, scored))

        infos = [{} for _ in range(self.num_envs)]
        # a goal is a scored "point" for logging and league stats even when the
        # episode carries on
        for i in np.nonzero(scored | conceded)[0]:
            infos[i]["result"] = 1 if scored[i] else -1
            infos[i]["episode"] = {"r": float(self.point_reward[i]),
                                   "l": int(self.point_ticks[i])}
        self.point_reward[scored | conceded] = 0
        self.point_ticks[scored | conceded] = 0

        idx = np.nonzero(done)[0]
        if len(idx):
            terminal = self._obs[idx]
            for j, i in enumerate(idx):
                sid = int(self.scn[i])
                infos[i].setdefault("result", 0)
                infos[i].setdefault("episode", {"r": float(self.ep_reward[i]),
                                                "l": int(self.ticks[i])})
                infos[i]["scenario"] = sid
                infos[i]["success"] = bool(success[i])
                infos[i]["terminal_observation"] = terminal[j]
                if not (scored[i] or conceded[i]):
                    infos[i]["TimeLimit.truncated"] = True
                if not _CONT[sid]:
                    self.history[sid].append(bool(success[i]))
            self._maybe_unlock()
            self._restart(done)

        return self._obs.copy(), rew, done, infos

    def close(self):
        self.hosts.close()

    def get_attr(self, name, indices=None):
        return [getattr(self, name)] * self.num_envs

    def set_attr(self, name, value, indices=None):
        setattr(self, name, value)

    def env_method(self, name, *args, indices=None, **kwargs):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
