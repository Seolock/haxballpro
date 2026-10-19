"""경기를 JSON으로 녹화한다 — 눈으로 보기 위한 것.

  python -m hbrl.record --a checkpoints/ppo_final.zip --b baseline --games 4

프레임 간격은 에이전트 스텝 하나(30Hz)라서 30fps로 재생하면 실제 속도가 된다.
에이전트 `a`가 학습자이고, 그가 맡는 팀 색은 학습·평가와 똑같이 경기마다
새로 뽑힌다.
"""

import argparse
import json

from hbrl import obs as O
from hbrl.baseline import load_agent
from hbrl.vecenv import HaxballVecEnv


def record(a, b, games, seed):
    env = HaxballVecEnv(workers=1, rooms=1, seed=seed)
    env.opponents = [b]
    env.n_fixed = env.n_league = 1
    env.idx_goalie = env.idx_idle = 0
    o = env.reset()

    out, frames = [], []
    while len(out) < games:
        side = int(env.side[0])
        o, _, done, info = env.step(a.act(o))
        s = env.hosts.state[0]
        c = env.hosts.bytes[0]
        frames.append([round(float(x), 1) for x in s[:2]] +
                      [round(float(x), 1) for x in s[4:6]] +
                      [round(float(x), 1) for x in s[8:10]] +
                      [int(c[0]), int(c[1]), side])
        if done[0]:
            out.append({"side": side, "result": info[0]["result"], "frames": frames})
            frames = []
    env.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="checkpoints/ppo_final.zip")
    ap.add_argument("--b", default="baseline")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="match.json")
    args = ap.parse_args()

    games = record(load_agent(args.a), load_agent(args.b), args.games, args.seed)
    data = {
        "a": args.a.split("/")[-1].replace(".zip", ""),
        "b": args.b.split("/")[-1].replace(".zip", ""),
        "pitch": {"halfW": O.HALF_W, "halfH": O.HALF_H, "goalX": O.GOAL_X,
                  "postY": O.GOAL_POST_Y, "ballR": O.BALL_R, "playerR": O.PLAYER_R,
                  "ballHalfH": 170.0},
        "games": games,
    }
    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    n = sum(len(g["frames"]) for g in games)
    print(f"{len(games)} games, {n} frames -> {args.out}")
    print("results:", [g["result"] for g in games])


if __name__ == "__main__":
    main()
