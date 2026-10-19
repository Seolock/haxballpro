// K sandbox rooms in one process, driven over stdin/stdout with fixed-size
// binary frames. Physics only: Python owns obs, reward and episode logic.
//
//   request  (K*3 bytes)  per room: [flag, redInput, blueInput]
//                                   flag 0 = step, 1 = kick-off reset, 2 = read only,
//                                   3 = scenario reset -- then byte 1 is the
//                                   scenario id and byte 2 is the mirror flag
//                                   (the input bytes are idle on a reset frame,
//                                   so this costs no extra protocol width)
//   response (K*51 bytes)          [K*12 float32][K*3 uint8]
//                                   floats: ball(x,y,vx,vy) red(...) blue(...)
//                                   bytes:  redCharged, blueCharged, event
//                                   event 0 = none, 1 = red goal, 2 = blue goal
//
// A `step` advances SKIP physics ticks with the inputs held constant -- the
// frame-skip the trainer used to do with SKIP separate round-trips. The score
// diff picked up by readInto still catches a goal anywhere in the window (a
// second goal can't happen: the first drops the room into BEFORE_KICKOFF).
const { makeRoom } = require("./room.js");

const K = parseInt(process.argv[2] || "64", 10);
const SEED = parseInt(process.argv[3] || "1", 10);
const SKIP = Math.max(1, parseInt(process.argv[4] || "1", 10));

const REQ = K * 3;
const rooms = [];
for (let i = 0; i < K; i++) rooms.push(makeRoom());

const res = Buffer.allocUnsafe(K * 51);
const f32 = new Float32Array(res.buffer, res.byteOffset, K * 12);
const u8 = new Uint8Array(res.buffer, res.byteOffset + K * 48, K * 3);

let seed = SEED >>> 0;
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);

function handle(req) {
  for (let i = 0; i < K; i++) {
    const r = rooms[i], o = i * 3;
    const flag = req[o];
    if (flag === 1) {
      r.reset(rnd() < 0.5);
      r.jitter(rnd, rnd);
    } else if (flag === 3) {
      r.setScenario(req[o + 1], rnd, req[o + 2] !== 0);
    } else if (flag === 0) {
      r.input(1, req[o + 1]);
      r.input(2, req[o + 2]);
      for (let k = 0; k < SKIP; k++) r.step();
    }
    r.readInto(f32, i * 12, u8, i * 3);
  }
  process.stdout.write(res);
}

let pending = Buffer.alloc(0);
process.stdin.on("data", (chunk) => {
  pending = pending.length ? Buffer.concat([pending, chunk]) : chunk;
  while (pending.length >= REQ) {
    handle(pending.subarray(0, REQ));
    pending = pending.subarray(REQ);
  }
});
process.stdin.on("end", () => process.exit(0));
