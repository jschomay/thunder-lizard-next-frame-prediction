import { HeadlessGame } from "./headless"
for (const numDinos of [0, 60]) {
  let t0 = performance.now()
  const g = new HeadlessGame({ seed: 7, numDinos, lava: true })
  let t1 = performance.now()
  for (let i = 0; i < 200; i++) g.step("NONE")
  let t2 = performance.now()
  for (let i = 0; i < 200; i++) g.observe()
  let t3 = performance.now()
  console.log({ numDinos, setupMs: Math.round(t1 - t0), stepMs: ((t2 - t1) / 200).toFixed(2), observeMs: ((t3 - t2) / 200).toFixed(2) })
  g.dispose()
}
