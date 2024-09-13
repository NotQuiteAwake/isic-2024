---
title: GPU providers
date: 01/08/24
author: Jimmy
colorlinks: true
---

## Providers

### CSD3 Wilkes

- 4xA100 nodes
- Documentation [here](https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html?highlight=gpu)
- SL3 has some level of access

> In addition it is possible to submit low priority jobs of no more than
12 hours duration to a special project project-sl4-type (where type=cpu or gpu)
which allows a limited number of jobs to run when the alternative is leaving
nodes idle.

### Kaggle

- Access via [API](https://www.kaggle.com/docs/api)
- Free 30 hours per user per week [link](https://www.kaggle.com/discussions/general/108481)
- [GPU is P100](https://www.kaggle.com/docs/efficient-gpu-usage)
- Reasonable performance vs. v100 [link](https://www.xcelerit.com/computing-benchmarks/insights/benchmarks-deep-learning-nvidia-p100-vs-v100-gpu/)

### Lambdalabs

- [pricing](https://lambdalabs.com/service/gpu-cloud#pricing)
- SkyPilot might be useful here
- 8x V100 for 4.4
- 8x A100 for 14
- 1x A100 for 1.29
- Non-negligible storage fees, 1000G is 200USD per month

### vast.ai

- Useful github [guide](https://github.com/joystiller/vast-ai-guide)
- [Docs](https://vast.ai/docs/cli/quickstart)
- Single V100 for 0.14
- 8x V100 for 1.9
- 8x A100 for 10
- Storage also has a (negligible) charge
- Supports moving data between instances
- However could have security and performance issues
- Interruptible instances don't look much cheaper for some reason.

### Massed Compute

- Appears to be a relatively new player
- Coupon RedditDeepLearning to get A6000 at 0.31
- A6000 [benchmarks](https://lambdalabs.com/blog/nvidia-rtx-a6000-benchmarks)

### GCP

- Has 300USD free credit, which can support about 100 hours of V100 or 50 hours
  2xV100 for a regular instance
- This and Colab both seem to have availability issues where GPUs better than
  V100 or even V100 itself might be unavailable.

### TPUs

Somewhat risky due to:

- Potential support and performance issues, especially on `PyTorch`
- Probably only available on expensive providers AWS/GCP
- [This article](https://arxiv.org/abs/2309.07181) shows high failure rate and
  significant slowdown of PyTorch functions under TPU

### Honourable mentions

- TensorDock (A100 at 1.6)
- Hyperstack (2.3)
- Runpod (A100 at 1.9)
- Modal (3.1 for 40G A100...)

## Setup

- Remember to update weights to disk every few epochs
- Consider `rrsync` to CSD3 RDS. Check `quota`. See [this](https://serverfault.com/a/965929)

