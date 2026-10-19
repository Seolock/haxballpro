"""js/play.js를 위한 추론 소켓 — 실전 대국의 Python 쪽 절반.

play.js가 매 게임 틱마다 raw 디스크 상태를 보내고, 돌아온 입력 마스크를
`setKeyState`로 누른다. 관측은 정책이 학습할 때 쓴 `obs.py` 코드를 그대로
태워서 만들므로, 학습과 실전 사이에 특징이 어긋날 여지가 없다.

  요청  51바이트: [12 float32 상태][u8 redCharged][u8 blueCharged][u8 side]
  응답   1바이트: HaxBall 입력 마스크
"""

import argparse
import os
import socket

import numpy as np
from stable_baselines3 import PPO

from hbrl import obs as O

REQ_SIZE = 12 * 4 + 3


def serve(model, path, deterministic):
    if os.path.exists(path):
        os.unlink(path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    print(f"[serve] listening on {path}")
    while True:
        conn, _ = srv.accept()
        print("[serve] client connected")
        kick_prev = np.zeros(1, dtype=np.float32)
        try:
            with conn:
                buf = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while len(buf) >= REQ_SIZE:
                        frame, buf = buf[:REQ_SIZE], buf[REQ_SIZE:]
                        state = np.frombuffer(frame, np.float32, 12).reshape(1, 12)
                        chg = np.frombuffer(frame, np.uint8, 2, 48).reshape(1, 2)
                        side = np.array([frame[50]], dtype=np.int8)
                        o = O.build_obs(state, np.pad(chg, ((0, 0), (0, 1))), side, kick_prev)
                        a, _ = model.predict(o, deterministic=deterministic)
                        a = np.asarray(a).reshape(-1)
                        kick_prev[:] = O.kick_bit(a)
                        conn.sendall(bytes([int(O.action_to_input(a, side)[0])]))
        except (ConnectionResetError, BrokenPipeError):
            pass
        print("[serve] client gone")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/ppo_final.zip")
    ap.add_argument("--socket", default="/tmp/hbrl.sock")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()
    model = PPO.load(args.model, device="cpu")
    serve(model, args.socket, args.deterministic)


if __name__ == "__main__":
    main()
