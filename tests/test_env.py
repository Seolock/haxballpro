import numpy as np
import pytest

from hbrl import obs as O
from hbrl.baseline import GoalieAgent, IdleAgent, RandomAgent, ScriptedAgent
from hbrl import scenarios as S
from hbrl.vecenv import GOAL_REWARD, TIME_PENALTY, HaxballVecEnv


def rollout(env, agent, n):
    o = env.reset()
    out = []
    for _ in range(n):
        o, r, d, info = env.step(agent.act(o))
        out.append((o.copy(), r.copy(), d.copy()))
    return out


def test_determinism():
    """Same seed + same actions must reproduce the trajectory exactly."""
    runs = []
    for _ in range(2):
        env = HaxballVecEnv(workers=1, rooms=8, seed=7)
        runs.append([x[0] for x in rollout(env, ScriptedAgent(seed=1), 120)])
        env.close()
    for a, b in zip(*runs):
        assert np.array_equal(a, b)


def test_mirror_symmetry():
    """A left-right mirrored world must give the mirrored side the same obs."""
    rng = np.random.default_rng(0)
    s = rng.normal(0, 100, (16, 12)).astype(np.float32)
    chg = rng.integers(0, 2, (16, 3)).astype(np.uint8)
    kp = rng.integers(0, 2, 16).astype(np.float32)

    red = O.build_obs(s, chg, np.zeros(16, np.int8), kp)

    m = s.copy()                                     # mirror x, swap the colours
    for off in (O.BALL, O.RED, O.BLUE):
        m[:, off] *= -1
        m[:, off + 2] *= -1
    m[:, O.RED:O.RED + 4], m[:, O.BLUE:O.BLUE + 4] = \
        m[:, O.BLUE:O.BLUE + 4].copy(), m[:, O.RED:O.RED + 4].copy()
    mc = chg.copy()
    mc[:, [0, 1]] = chg[:, [1, 0]]
    blue = O.build_obs(m, mc, np.ones(16, np.int8), kp)

    assert np.allclose(red, blue, atol=1e-5)


def test_action_mask_mirror():
    """Blue's inputs are the left-right mirror of red's for the same action."""
    a = np.arange(O.N_ACTIONS)
    r = O.action_to_input(a, np.zeros(O.N_ACTIONS, np.int8))
    b = O.action_to_input(a, np.ones(O.N_ACTIONS, np.int8))
    keep = np.uint8(255 - (O.LEFT | O.RIGHT))
    swap = lambda m: (m & keep) | ((m & O.LEFT) << 1) | ((m & O.RIGHT) >> 1)
    assert np.array_equal(swap(r), b)


def test_goal_reward_sign():
    """Every scored goal pays +GOAL_REWARD, every concession -GOAL_REWARD."""
    env = HaxballVecEnv(workers=1, rooms=16, seed=3)
    env.opponents = [IdleAgent()]
    env.n_fixed = env.n_league = 1
    env.idx_goalie = env.idx_idle = 0
    o = env.reset()
    agent = ScriptedAgent(epsilon=0.0)
    goals = 0
    for _ in range(900):
        o, r, d, info = env.step(agent.act(o))
        for i in np.nonzero(d)[0]:
            res = info[i]["result"]
            if res:
                goals += 1
                assert r[i] == pytest.approx(res * GOAL_REWARD - TIME_PENALTY, abs=1e-4)
    env.close()
    assert goals > 0


def test_scripted_beats_random():
    env = HaxballVecEnv(workers=2, rooms=32, seed=5)
    env.opponents = [RandomAgent(seed=11)]
    env.n_fixed = env.n_league = 1
    env.idx_goalie = env.idx_idle = 0
    env.reset()
    env.opp_idx[:] = 0
    o, wins, losses = env._obs, 0, 0
    agent = ScriptedAgent(seed=2)
    for _ in range(1200):
        o, r, d, info = env.step(agent.act(o))
        for i in np.nonzero(d)[0]:
            wins += info[i]["result"] == 1
            losses += info[i]["result"] == -1
        env.opp_idx[:] = 0
    env.close()
    assert wins > 3 * losses, f"scripted {wins} : random {losses}"


def test_every_scenario_runs():
    """Each scenario lays out, ends inside its own limit, and reports a verdict."""
    env = HaxballVecEnv(workers=1, rooms=16, seed=1)
    agent = ScriptedAgent(epsilon=0.0)
    for sc in S.SCENARIOS:
        if sc.continuous:
            continue                      # 2 minutes is too long for a unit test
        env.pinned = sc.id
        o = env.reset()
        assert np.array_equal(env.scn, np.full(env.num_envs, sc.id))
        seen = 0
        for _ in range(sc.limit + 2):
            o, r, d, info = env.step(agent.act(o))
            for i in np.nonzero(d)[0]:
                assert info[i]["scenario"] == sc.id
                assert isinstance(info[i]["success"], bool)
                seen += 1
        assert seen >= env.num_envs, f"{sc.name} did not finish within its limit"
    env.close()


def test_curriculum_unlocks_on_success():
    """A scenario whose window fills with wins unlocks the next one."""
    env = HaxballVecEnv(workers=1, rooms=8, seed=2, curriculum=True)
    assert env.unlocked == 1
    env.history[0].extend([True] * S.WINDOW)
    env._maybe_unlock()
    assert env.unlocked == 2
    env.history[1].extend([False] * S.WINDOW)
    env._maybe_unlock()
    assert env.unlocked == 2, "a failing window must not unlock anything"
    env.close()


def test_pool_thinning_keeps_opponents_valid():
    """Thinning the league must not leave an env pointing at a removed agent.

    A stale index matches no branch in `_opp_actions`, so the opponent falls back
    to action 0 and jogs into a corner for the rest of the episode -- free wins
    that inflate the win rate and teach nothing.
    """
    from hbrl.baseline import IdleAgent
    from hbrl.vecenv import MAX_POOL

    env = HaxballVecEnv(workers=1, rooms=32, seed=4)
    env.league = True
    for _ in range(MAX_POOL + 8):
        env.add_snapshot(IdleAgent())
        assert env.opp_idx.max() < len(env.opponents)

    env.opp_idx[:] = len(env.opponents) - 1        # everyone on the newest snapshot
    env.add_snapshot(IdleAgent())                  # ... which thinning may drop
    assert env.opp_idx.max() < len(env.opponents)
    env.close()
