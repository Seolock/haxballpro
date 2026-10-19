# haxballpro

[HaxBall](https://www.haxball.com) 1v1 강화학습 에이전트.

## 설치

```bash
cd haxballpro
conda create -n hbrl python=3.12 nodejs -y
conda activate hbrl
pip install -r requirements.txt
cd js && npm install && cd ..
```

설치 확인

```bash
pytest tests/ -q            # -> "8 passed"
node js/bench.js 1 200000   # -> "... -> Nk env-steps/s"
```

## 학습

`hbrl/scenarios.py`의 시나리오를 순서대로 하나씩 클리어하며 학습한다 — 빈 골대
마무리 → 운반 → 골키퍼 뚫기 → 경합 → 수비 → 걷어내기 → 킥오프. 성공률 70%를
넘기거나 스텝 예산을 쓰면 다음 시나리오가 해금되고, 전부 풀면 리셋 없는
`full_game`으로 넘어간다. 보상은 골 ±10과 시간 패널티뿐이고 보조 보상은 없다.

학습이 며칠 걸리므로 `screen` 세션 안에서 띄워 SSH가 끊겨도 살아남게 한다:

```bash
screen -S hb                                    # 세션 생성 후 진입
conda activate hbrl
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 python -m hbrl.train --workers 16 --rooms 48
```

```bash
python -m hbrl.train --resume     # 마지막 체크포인트부터 이어서
tensorboard --logdir runs
```

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--workers` × `--rooms` | 8 × 64 | 병렬 환경 수. workers는 코어 수보다 2 적게 |
| `--steps` | 1B | 총 학습 스텝 |
| `--warmup-steps` | 20M | 스크립트 상대 구간. 이후 자기대전 리그가 켜진다 |
| `--snapshot-every` | 10M | 체크포인트 저장 = 리그에 과거 자신을 추가 |
| `--n-steps` | 512 | 롤아웃 길이. 학습 dynamics엔 무해, 크면 스텝당 오버헤드가 준다 |
| `--batch-size` | 2048 | 키우지 말 것 |
| `--sync` | off | 파이프라인 수집기 끄기 |


## 경기 보기

```bash
python -m hbrl.record --a checkpoints/ppo_final.zip --b goalie --games 6
```

경기를 `match.json`으로 녹화한다. 상대는 `baseline` / `goalie` / `random` / `idle`
또는 다른 체크포인트 경로.

**모델이 얼마나 센지는 사람과 붙여 봐야 안다.** 스크립트 상대와의 대전표는
실력을 재지 못한다 — 직전 1B 모델이 규칙 기반 봇을 161-0으로 이기면서 골대 앞에
서 있기만 하는 상대에게는 20판에 1골밖에 못 넣었다. 학습이 진행되는지는
TensorBoard의 `curriculum/success_*`로 보고, 실제 실력은 아래 방법으로 직접
붙어서 확인한다.

## 실제 게임에서 플레이하기

**1.** https://www.haxball.com 에서 *Create Room*, 맵은 반드시 **Classic**으로.
정책은 그 경기장 치수로 관측을 정규화하므로 다른 맵은 모른다. 방 ID는 링크의
`c=` 파라미터다 — `haxball.com/play?c=31IBNI3w4F0` 이면 `31IBNI3w4F0`.

**2.** 정책 서버를 띄운다.

```bash
python -m hbrl.serve --model checkpoints/ppo_final.zip
```

**3.** 다른 터미널에서 에이전트를 접속시킨다.

```bash
node js/play.js 31IBNI3w4F0 hbrl        # <roomId> [playerName] [socketPath]
```

`[play] left: died at ConnectingToPeer` 가 뜨면 이 머신의 NAT/방화벽이 HaxBall
P2P를 막는 것이다 (기관·클라우드 서버가 흔히 그렇다. node-haxball엔 TURN 폴백이
없다). `play.js`를 일반 가정용 회선 머신(노트북 등)에서 돌려라 —
`checkpoints/ppo_final.zip`(~2MB)만 복사하면 `serve.py`도 거기서 CPU로 띄울 수
있다.

**4.** 에이전트를 팀에 넣는다. 관중석에서는 가만히 서 있는다. 반대 팀에 직접
들어가 게임을 시작하면 된다. 골이 날 때마다 콘솔에 찍힌다.

안 움직인다면 셋 중 하나이고, 이유가 콘솔에 나온다: 아직 관중석, 맵이 Classic이
아님, 필드에 1v1이 아닌 인원.

**본인이 만든 방에서 쓸 것.** 공개방에 봇을 넣으면 다른 사람들에게 영향을 준다.
