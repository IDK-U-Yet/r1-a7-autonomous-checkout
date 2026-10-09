# R1-A7 Collection Console

Read-only desktop dashboard for the four-camera cashier collection workflow. It
subscribes to `teleimager`, reads the managed collection state, scanner audit
log, and collector diagnostics, and never opens V4L2 devices or publishes robot
commands.

## Build

```bash
cd /home/user/r1-a7-autonomous-checkout/apps/r1-collection-console/frontend
pnpm install
pnpm lint
pnpm build
```

The backend requires `aiohttp`, which is already present in `tv_official`.

## Run

```bash
export TASK_NAME=cashier_oreo_16d_v1_80
TV_PYTHON=/home/user/miniconda3/envs/tv_official/bin/python \
  /home/user/r1-a7-autonomous-checkout/scripts/start-collection-console.sh \
  --task-name "$TASK_NAME"
```

Open <http://127.0.0.1:8765/> on the collection workstation.
