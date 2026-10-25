---
license: apache-2.0
tags:
- reinforcement-learning
- atari
- tensorflowjs
---

# BBF on Atari 100k, for TensorFlow.js

Snapshots of BBF ("Bigger, Better, Faster", Schwarzer et al. 2023) trained on Atari 100k with the released `BBF.gin`
(replay ratio 64, 2 updates per move), one seed per game, saved every 10,000 moves. Used by the page
[Two Hours of Atari](https://elichen.github.io/atari/).

`<game>/<step>/policy.json` describes the tensors in `policy.bin` (target network; conv kernels float16,
dense kernels int8 with one scale per output column). `test.json` holds reference Q values from JAX on real game states.
