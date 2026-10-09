# R1-A7 dataset shadow inference

The R1-A7 policy and dataset action contract is fixed at 16 dimensions:

```text
[left arm 7, right arm 7, left gripper 1, right gripper 1]
```

Raw demonstrations may still contain `waist.qpos` for audit. Conversion excludes
that field from model state and action. The real-robot bridge holds waist yaw
outside the policy contract. Existing 17D datasets and checkpoints are not
compatible with this runtime and must be retrained after conversion.

Validate a local LeRobot v3 dataset frame without loading a policy:

```bash
conda run -n unitree_lerobot python -m unitree_lerobot.r1a7_shadow_inference \
  --dataset-root /home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30 \
  --episode 0 \
  --frame 0 \
  --dry-run
```

Run policy inference on the same recorded observation, with no publishers or
robot controller imports:

```bash
conda run -n unitree_lerobot python -m unitree_lerobot.r1a7_shadow_inference \
  --dataset-root /home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30 \
  --episode 0 \
  --frame 0 \
  --task "Pick up a product from the blue tray on the left, present it to the scanner in the scanning area, and wait until the scan is accepted." \
  --policy-path /absolute/path/to/checkpoint/pretrained_model \
  --device cuda
```

`--frame` is relative to the selected episode. The command validates LeRobot
v3 metadata, the exact joint order, all four camera keys, action shape, finite
values, and R1-A7/Dex1 position ranges before emitting JSON. It has no
real-robot execution mode. The separate hardware gateway remains locked unless
two explicit authorization values are supplied, and resolving controller types
still does not instantiate a controller or create a publisher.

## Export a latest PI0.5 checkpoint for the legacy runtime

Only full checkpoints without PEFT, memory, relative actions, or RTC can be
represented by the legacy policy implementation. Export one without loading
the model weights:

```bash
conda run -n unitree_lerobot python -m unitree_lerobot.export_pi05_legacy \
  --source-policy /absolute/path/to/latest/checkpoint/pretrained_model \
  --dataset-root /home/jieguo/.cache/huggingface/lerobot/local/r1a7_black_truffle_4cam_30 \
  --output /absolute/path/to/legacy/checkpoint \
  --weights-mode hardlink \
  --device cuda \
  --tokenizer-name /absolute/path/to/local/paligemma-tokenizer
```

Use `--weights-mode copy` when input and output are on different filesystems.
The output directory must not already exist. The exporter filters config fields
using the legacy `PI05Config` dataclass, rebuilds both processors from dataset
metadata and statistics, then parses the result with the legacy config parser.
