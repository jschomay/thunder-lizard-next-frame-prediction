// Lets harness/ scripts import the game's dependencies (one shared copy of each).
import { resolve } from "path"

const game = (p: string) => resolve(__dirname, "game/node_modules", p)

export default {
  resolve: {
    alias: {
      bitecs: game("bitecs"),
      "@timohausmann/quadtree-ts": game("@timohausmann/quadtree-ts"),
    },
  },
}
