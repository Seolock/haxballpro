// The trained agent playing in a real HaxBall room.
//
// Joins as an ordinary client and drives the keyboard with Utils.keyState(),
// exactly like a human -- no disc teleporting. Each game tick it ships the raw
// disc state to hbrl/serve.py and applies the mask that comes back. The reply
// lands one frame later; that lag is deliberate and matches the observation
// delay the policy is trained against.
//
//   node play.js <roomId> [playerName] [socketPath]
const net = require("net");
const API = require("node-haxball")();
const { Utils, Room } = API;

const ROOM_ID = process.argv[2];
const NAME = process.argv[3] || "hbrl";
const SOCK = process.argv[4] || "/tmp/hbrl.sock";

// node-haxball ConnectionState enum
const CONN = { "-1": "TryingReverseConnection", 0: "ConnectingToMaster",
  1: "ConnectingToPeer", 2: "AwaitingState", 3: "Active", 4: "ConnectionFailed" };
if (!ROOM_ID) {
  console.error("usage: node play.js <roomId> [playerName] [socketPath]");
  process.exit(1);
}

const req = Buffer.alloc(51);
const f32 = new Float32Array(req.buffer, req.byteOffset, 12);
let mask = 0;                                  // latest action from the policy
let inflight = false;

const sock = net.createConnection(SOCK);
sock.on("data", (d) => { mask = d[d.length - 1]; inflight = false; });
sock.on("error", (e) => { console.error("[play] inference socket:", e.message); process.exit(1); });

const live = (p) => p.disc && (p.disc.ext || p.disc);

// The policy only knows Classic: obs.py normalises by that pitch's dimensions,
// so on any other stadium the observation is meaningless. Sit still instead.
const STADIUM = "Classic";
let warned = null;

function tick(room) {
  room.extrapolate();                            // work on the same frame the game draws
  const me = room.currentPlayer;
  const gs = room.gameStateExt || room.gameState;
  if (!gs || !me || !me.team || me.team.id === 0 || !live(me)) return;

  const name = gs.stadium && gs.stadium.name;
  if (name !== STADIUM) {
    if (warned !== name) {
      warned = name;
      console.error(`[play] stadium is "${name}", agent only plays "${STADIUM}" -- idling`);
    }
    room.setKeyState(0);
    return;
  }

  // 1v1 only: exactly one other player on the pitch.
  const opp = room.state.players.find((p) => p.id !== me.id && p.team && p.team.id && live(p));
  if (!opp) return;

  const ball = gs.physicsState.discs[0];
  const side = me.team.id === 2 ? 1 : 0;        // 1 = blue
  const red = side ? opp : me, blue = side ? me : opp;

  const put = (i, d) => {
    f32[i] = d.pos.x; f32[i + 1] = d.pos.y; f32[i + 2] = d.speed.x; f32[i + 3] = d.speed.y;
  };
  put(0, ball); put(4, live(red)); put(8, live(blue));
  req[48] = red.isKicking ? 1 : 0;
  req[49] = blue.isKicking ? 1 : 0;
  req[50] = side;

  if (!inflight) { inflight = true; sock.write(req); }
  room.setKeyState(mask);
}

Utils.generateAuth().then(([authKey, authObj]) => {
  Room.join({ id: ROOM_ID, authObj }, {
    // crappy_router: 10s join timeout instead of 4s -- WebRTC negotiation from a
    // server is slower than from a browser and often misses the 4s window.
    storage: { player_name: NAME, avatar: "AI", player_auth_key: authKey, crappy_router: true },
    noPluginMechanism: true,
    onConnInfo: (state, extra) => {
      console.log("[play] conn:", CONN[state] || state,
        state === 4 ? "-- " + String(extra) : "");
    },
    onOpen: (room) => {
      console.log("[play] joined", ROOM_ID, "as", NAME);
      room._onGameTick = () => tick(room);
      room._onTeamGoal = (team) => {
        const gs = room.gameState;
        console.log(`[play] goal by ${team === 1 ? "red" : "blue"} -- ${gs.redScore}:${gs.blueScore}`);
      };
    },
    onClose: (err) => {
      console.log("[play] left:", err === undefined
        ? "died at ConnectingToPeer -- master handshake was fine (room id is valid), the WebRTC "
          + "connection to the host browser failed. Almost always a NAT/firewall on this host: "
          + "run play.js from a machine on normal consumer NAT instead (see README)."
        : String(err));
      process.exit(0);
    },
  });
});
