"""시나리오 커리큘럼 표 — 학습이 거쳐야 할 상황들의 단일 정의.

Google Research Football의 academy 시나리오와 같은 발상이다. 상태 분포가 저절로
좋아지길 기다리지 않고, 필요한 상황을 명시적으로 만들어 하나씩 클리어시킨다.
직전 학습(1.02B 스텝)이 실패한 이유가 정확히 이것이다 — 모든 에피소드가
킥오프에서 시작해 20초에 잘려서, 수비 상황이 전체의 1%, 킥 사거리 안이 3.2%밖에
안 나왔다.

여기에는 id·제한시간·상대·판정 규칙만 있다. 디스크를 실제로 어디에 놓을지는
엔진 관심사라 `js/room.js: layout()`이 갖는다. 두 곳이 공유하는 건 id 숫자뿐이다.
"""

from dataclasses import dataclass

FPS = 30                                    # agent steps per second (frame_skip 2)


@dataclass(frozen=True)
class Scenario:
    id: int
    name: str
    limit: int                              # agent steps before the episode is cut
    opponent: str                           # "idle"|"scripted"|"goalie"|"keeper"|"league"
    defend: bool = False                    # success = survive without conceding
    clear: bool = False                     # success = get the ball past midfield
    kickoff: bool = False                   # start from a kick-off, not a layout
    continuous: bool = False                # goals do not end the episode


SCENARIOS = [
    # --- 공격 --------------------------------------------------------------
    Scenario(0, "empty_goal_close", 5 * FPS, "idle"),
    Scenario(1, "empty_goal", 10 * FPS, "idle"),
    Scenario(2, "shoot_with_keeper", 10 * FPS, "keeper"),
    Scenario(3, "contest", 15 * FPS, "league"),
    # --- 수비 --------------------------------------------------------------
    # 리그 상대로 두면 초반엔 공격수가 무능해서 가만히 있어도 성공한다 (무작위
    # 51.6% / 70.3% 측정). 유능한 스크립트 공격수를 고정으로 붙여 난이도를 고정한다.
    Scenario(4, "defend", 10 * FPS, "scripted", defend=True),
    Scenario(5, "clearance", 10 * FPS, "scripted", defend=True, clear=True),
    # --- 실제 경기 ---------------------------------------------------------
    Scenario(6, "kickoff", 20 * FPS, "league", kickoff=True),
    # 커리큘럼을 마친 뒤의 최종 단계. 골이 나도 에피소드가 끝나지 않고 엔진이
    # AfterGoal -> 상대 킥오프로 이어가므로, 20초 절단이 잘라먹던 긴 공방과
    # 실점 직후 수비가 비로소 데이터에 들어온다. 2분은 무한루프 방지 + 진영과
    # 리그 상대를 다시 뽑기 위한 안전장치일 뿐이다.
    Scenario(7, "full_game", 120 * FPS, "league", kickoff=True, continuous=True),
]

CURRICULUM = [s for s in SCENARIOS if not s.continuous]     # 0..6, 순서대로 해금
FULL_GAME = SCENARIOS[-1]

LIMITS = [s.limit for s in SCENARIOS]
UNLOCK_RATE = 0.7                           # 이 성공률을 넘기면 다음 시나리오 해금
WINDOW = 400                                # 성공률을 재는 최근 에피소드 수

# 성공률만으로 게이팅하면 천장이 낮은 시나리오에서 커리큘럼이 영원히 멈춘다.
# 예산을 쓰면 못 넘겨도 다음으로 넘어간다 — 해금된 시나리오는 계속 섞여 나오므로
# 버려지는 게 아니라 병행 학습으로 바뀔 뿐이다.
UNLOCK_BUDGET = 30_000_000                  # 한 시나리오에 쓸 수 있는 최대 env 스텝
