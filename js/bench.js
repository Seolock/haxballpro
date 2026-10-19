// Phase 0 gate: how fast can one node process step node-haxball sandbox rooms?
const { makeRoom } = require("./room.js");

const K = parseInt(process.argv[2] || "1", 10);       // rooms in this process
const STEPS = parseInt(process.argv[3] || "200000", 10);

const rooms = [];
for (let i = 0; i < K; i++) rooms.push(makeRoom());
const f32 = new Float32Array(12 * K), u8 = new Uint8Array(3 * K);

let seed = 12345;
const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

const t0 = process.hrtime.bigint();
for (let s = 0; s < STEPS; s++) {
  for (let i = 0; i < K; i++) {
    const r = rooms[i];
    r.input(1, (rnd() * 32) | 0);
    r.input(2, (rnd() * 32) | 0);
    r.step();
    r.readInto(f32, 12 * i, u8, 3 * i);                 // include state extraction cost
  }
}
const dt = Number(process.hrtime.bigint() - t0) / 1e9;
const total = STEPS * K;
console.log(`rooms=${K} steps=${total} time=${dt.toFixed(2)}s -> ${(total / dt / 1000).toFixed(1)}k env-steps/s`);
console.log("sample state:", Array.from(f32.slice(0, 12)).map((x) => +x.toFixed(2)));
