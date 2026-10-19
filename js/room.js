// One 1v1 HaxBall sandbox room, driven step by step. Physics only -- no obs,
// no reward, no episode logic; Python owns all of that.
const { Room, Utils } = require("node-haxball")();

const RED = 1, BLUE = 2;
const RED_ID = 1, BLUE_ID = 2;

// event byte returned by read()
const EV_NONE = 0, EV_RED_GOAL = 1, EV_BLUE_GOAL = 2;

// GamePlayState
const BEFORE_KICKOFF = 0, PLAYING = 1;

function makeRoom(stadium = "Classic") {
  // No animation frames: nothing must tick unless we call runSteps().
  const room = Room.sandbox({}, {
    requestAnimationFrame: () => 0,
    cancelAnimationFrame: () => {},
  });
  room.setSimulationSpeed(0);

  room.setCurrentStadium(defaultStadium(stadium), 0);
  room.setTimeLimit(0, 0);
  room.setScoreLimit(0, 0);

  room.playerJoin(RED_ID, "R", "", "", "", "");
  room.playerJoin(BLUE_ID, "B", "", "", "", "");
  room.setPlayerTeam(RED_ID, RED, 0);
  room.setPlayerTeam(BLUE_ID, BLUE, 0);
  room.startGame(0);

  let red = room.state.players.find((p) => p.id === RED_ID);
  let blue = room.state.players.find((p) => p.id === BLUE_ID);

  let prevRed = 0, prevBlue = 0;

  // Three snapshots, all taken once at construction by scripting a red goal:
  //   kickoffs[0]  red kicks off      (the state the room is created in)
  //   kickoffs[1]  blue kicks off     -- so episodes do not always start with
  //                red on the ball, which hands one colour a standing advantage
  //   playing      mid-play           -- no kick-off fence, so scenario setups
  //                can put the discs anywhere and both players can reach them
  const kickoffs = [room.takeSnapshot(), null];
  let playing = null;
  for (let i = 0; i < 3000 && !kickoffs[1]; i++) {
    room.playerInput(8 | 16, RED_ID);
    room.playerInput(0, BLUE_ID);
    room.runSteps(1);
    if (!playing && room.gameState.state === PLAYING) playing = room.takeSnapshot();
    if (room.gameState.redScore > 0 && room.gameState.state === BEFORE_KICKOFF)
      kickoffs[1] = room.takeSnapshot();
  }
  if (!kickoffs[1] || !playing) throw new Error("could not reach kick-off / playing states");
  room.useSnapshot(kickoffs[0]);
  red = room.state.players.find((p) => p.id === RED_ID);
  blue = room.state.players.find((p) => p.id === BLUE_ID);

  const self = {
    room,
    input(id, mask) { room.playerInput(mask & 31, id); },
    step() { room.runSteps(1); },

    // Writes 12 floats + 3 bytes. Returns the event byte.
    readInto(f32, fi, u8, ui) {
      const gs = room.gameState;
      const discs = gs.physicsState.discs;
      const b = discs[0], r = red.disc, l = blue.disc;
      f32[fi + 0] = b.pos.x;  f32[fi + 1] = b.pos.y;  f32[fi + 2] = b.speed.x;  f32[fi + 3] = b.speed.y;
      f32[fi + 4] = r.pos.x;  f32[fi + 5] = r.pos.y;  f32[fi + 6] = r.speed.x;  f32[fi + 7] = r.speed.y;
      f32[fi + 8] = l.pos.x;  f32[fi + 9] = l.pos.y;  f32[fi + 10] = l.speed.x; f32[fi + 11] = l.speed.y;
      u8[ui + 0] = red.isKicking ? 1 : 0;
      u8[ui + 1] = blue.isKicking ? 1 : 0;
      let ev = EV_NONE;
      if (gs.redScore !== prevRed) { ev = EV_RED_GOAL; prevRed = gs.redScore; }
      else if (gs.blueScore !== prevBlue) { ev = EV_BLUE_GOAL; prevBlue = gs.blueScore; }
      u8[ui + 2] = ev;
      return ev;
    },

    // Back to kick-off. Cheap: reuses the snapshot taken at construction.
    reset(which) {
      restore(kickoffs[which ? 1 : 0]);
    },

    // Drop the discs into a training scenario. Built on the mid-play snapshot,
    // so there is no kick-off fence and both players can reach the ball at once.
    // Layouts are written in red's frame (red attacks +x); `mirror` hands the
    // same situation to blue instead, which keeps both colours in the data.
    setScenario(id, rnd, mirror) {
      restore(playing);
      const s = layout(id, rnd);
      const put = (disc, p) => {
        disc.pos.x = mirror ? -p[0] : p[0];
        disc.pos.y = p[1];
        disc.speed.x = 0;
        disc.speed.y = 0;
      };
      put(ball(), s.ball);
      put(mirror ? blue.disc : red.disc, s.me);
      put(mirror ? red.disc : blue.disc, s.opp);
    },

    // Decorrelate episodes: nudge both players inside their own half. The ball
    // stays at the centre spot so the kick-off stays legal.
    jitter(rx, ry) {
      for (const p of [red, blue]) {
        p.disc.pos.x += (rx() - 0.5) * 80;
        p.disc.pos.y += (ry() - 0.5) * 160;
      }
    },

    get playing() { return room.gameState && room.gameState.state === PLAYING; },
    destroy() { room.destroy(); },
  };

  function ball() { return room.gameState.physicsState.discs[0]; }

  function restore(snap) {
    room.useSnapshot(snap);
    const ps = room.state.players;
    red = ps.find((p) => p.id === RED_ID);
    blue = ps.find((p) => p.id === BLUE_ID);
    // snapshots taken after a goal carry a non-zero score; baseline off it
    prevRed = room.gameState.redScore;
    prevBlue = room.gameState.blueScore;
  }

  return self;
}

// --- scenario layouts, in red's frame: red attacks +x, its own goal is at -x --
// `me` is the learner, `opp` the opponent. Positions are jittered by the caller's
// RNG so the agent cannot memorise one arrangement.
const GX = 370;                                   // goal line
const BEHIND = 40;                                // how far behind the ball to stand

function layout(id, rnd) {
  const j = (a) => (rnd() - 0.5) * 2 * a;         // uniform in [-a, a]
  switch (id) {
    case 0: {                                     // empty_goal_close
      const ball = [330 + j(15), j(40)];
      return { ball, me: [ball[0] - BEHIND, ball[1]], opp: [-300, 150] };
    }
    case 1: {                                     // empty_goal
      const ball = [j(40), j(60)];
      return { ball, me: [ball[0] - BEHIND, ball[1]], opp: [-320, j(120)] };
    }
    case 2: {                                     // shoot_with_keeper
      // 골대에서 멀면 골키퍼가 무조건 먼저 닿는다 (x=150일 때 스크립트조차 1.7%).
      // 붙여 놓고 y를 넓게 흔들어 각도를 만들어 준다.
      const ball = [250 + j(40), j(80)];
      const dx = GX - ball[0], dy = -ball[1];     // opponent goal is at +GX
      const n = Math.sqrt(dx * dx + dy * dy) || 1;
      return { ball, me: [ball[0] - BEHIND, ball[1]],
               opp: [GX - (dx / n) * 90, -(dy / n) * 90] };
    }
    case 3:                                       // contest
      return { ball: [0, j(40)], me: [-60 + j(20), j(40)], opp: [60 + j(20), j(40)] };
    case 4: {                                     // defend
      const ball = [-150 + j(40), j(60)];
      return { ball, me: [-260 + j(20), j(40)], opp: [ball[0] + BEHIND, ball[1]] };
    }
    case 5: {                                     // clearance
      const ball = [-300 + j(20), j(50)];
      return { ball, me: [ball[0] - 25, ball[1] + j(30)], opp: [ball[0] + BEHIND, ball[1]] };
    }
    default:
      throw new Error("no such scenario: " + id);
  }
}

function defaultStadium(name) {
  const s = Utils.getDefaultStadiums().find((x) => x.name === name);
  if (!s) throw new Error("no such default stadium: " + name);
  return s;
}

module.exports = { makeRoom };
