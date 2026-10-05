// Quick check: run a few ticks and print the view as ASCII.
import { HeadlessGame, VIEW } from "./headless"

const g = new HeadlessGame({ seed: 1, numDinos: 5, lava: true })
const T = ".~:,;\"#*%"  // ocean water dirt grass shrub jungle lava burnt
const show = () => {
  const [t, e] = g.observe()
  let s = ""
  for (let y = 0; y < VIEW; y++) {
    for (let x = 0; x < VIEW; x++) {
      const i = y * VIEW + x
      s += e[i] === 0 ? T[t[i]] : e[i] <= 20 ? "@" : e[i] === 41 ? "x" : "D"
    }
    s += "\n"
  }
  console.log(s)
}
show()
for (let i = 0; i < 30; i++) g.step("RIGHT")
show()
console.log("player", g.level.playerDino.getXY().toString(), "done:", g.done)
