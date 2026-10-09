#!/usr/bin/env bash
# Independent RTC experiment. This script is never called by collection launchers.
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
PI_PYTHON="${R1A7_PI05_PYTHON:-$ROOT_DIR/runtime/envs/pi05-262d2fd/bin/python}"
TV_PYTHON="${TV_PYTHON:-/home/user/miniconda3/envs/tv_official/bin/python}"
CHECKPOINT="$ROOT_DIR/runtime/checkpoints/pi05_oreo_v2v3_standard160_a100_gpu3_full_b24_lr1e5_8k_01/checkpoints/008000/pretrained_model"
DATASET="$ROOT_DIR/runtime/datasets/lerobot/r1a7_oreo_4cam_16d_v2v3_standard160_train"
NETWORK=enx00e04c187010
CONFIRMATION=I_UNDERSTAND_R1A7_COMMANDS_MOVE_REAL_HARDWARE
PRECHECK=0
CONFIRMED=""
EXTRA=()
while (($#)); do
  case "$1" in
    --preflight) PRECHECK=1; shift ;;
    --allow-task-override) EXTRA+=("$1"); shift ;;
    --help|-h)
      printf '%s\n' 'Independent RTC language test (default standard160 8K): --preflight (no hardware), --network-interface NAME,' \
        '  --policy-path DIR, --dataset-root DIR (select matching checkpoint/dataset),' \
        '  --confirm-real-robot I_UNDERSTAND_R1A7_COMMANDS_MOVE_REAL_HARDWARE' \
        '  --max-commands N (default 30; 0 unlimited), --execution-steps N (10),' \
        '  --prefetch-steps N (7), --initial-delay-steps N (7),' \
        '  --max-guidance-weight FLOAT (10), --device cuda, --image-host HOST,' \
        '  --num-inference-steps N (10; independent RTC sampling override),' \
        '  --rtc-mask-mode configured|overlap (configured: endpoint 10; overlap: H-s),' \
        '  --expected-barcode CODE, --task-before TEXT, --task-after TEXT, --output PATH.' \
        '  --allow-task-override (language experiment), --reference-task-before TRAINED_TEXT.' \
        '15Hz, no chunk interpolation. P pause, N resume, R return/restart, Q exit.' \
        'R requires empty grippers and a cleared return path. Ctrl-C exits.'
      exit 0 ;;
    --network-interface)
      (($# >= 2)) || exit 2
      NETWORK="$2"; shift 2 ;;
    --policy-path)
      (($# >= 2)) || exit 2
      CHECKPOINT="$2"; shift 2 ;;
    --dataset-root)
      (($# >= 2)) || exit 2
      DATASET="$2"; shift 2 ;;
    --confirm-real-robot)
      (($# >= 2)) || exit 2
      CONFIRMED="$2"; shift 2 ;;
    --max-commands|--execution-steps|--prefetch-steps|--initial-delay-steps|--max-guidance-weight|--num-inference-steps|--rtc-mask-mode|--device|--image-host|--expected-barcode|--task-before|--task-after|--reference-task-before|--output)
      (($# >= 2)) || exit 2
      EXTRA+=("$1" "$2"); shift 2 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
export PYTHONPATH="$ROOT_DIR/data/unitree-lerobot:$ROOT_DIR/runtime/lerobot-a100-262d2fd/src:$ROOT_DIR/robot/xr-teleoperate:$ROOT_DIR/robot/xr-teleoperate/teleop/teleimager/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ARGS=(--dataset-root "$DATASET" --policy-path "$CHECKPOINT" --tokenizer-path "$CHECKPOINT/tokenizer"
  --output "$ROOT_DIR/runtime/logs/rtc_language_$(date +%Y%m%d_%H%M%S_%N).jsonl"
  --confirm-real-robot "$CONFIRMATION" "${EXTRA[@]}")
# Validate parser and metadata before any DDS process can be started.
"$PI_PYTHON" - "${ARGS[@]}" <<'PY'
import json,sys,math
from pathlib import Path
from unitree_lerobot.r1a7_rtc_language import build_parser, read_training_tasks, resolve_task_contract
args=build_parser().parse_args(sys.argv[1:])
config=json.loads((args.policy_path/'config.json').read_text())
assert config['type']=='pi05'
assert config['input_features']['observation.state']['shape']==[16]
assert config['output_features']['action']['shape']==[16]
assert 0 < args.prefetch_steps < args.execution_steps < config['chunk_size']
assert 0 <= args.initial_delay_steps < config['chunk_size']
assert args.max_commands >= 0 and math.isfinite(args.max_guidance_weight) and args.max_guidance_weight > 0
assert args.num_inference_steps > 0, 'num-inference-steps must be positive'
for filename in ('model.safetensors','policy_preprocessor.json','policy_postprocessor.json','tokenizer/tokenizer.json'):
    if not (args.policy_path/filename).is_file():
        raise SystemExit(f'Missing checkpoint file: {filename}')
for filename in ('meta/info.json','meta/stats.json','meta/tasks.parquet'):
    if not (args.dataset_root/filename).is_file():
        raise SystemExit(f'Missing dataset file: {filename}')
info=json.loads((args.dataset_root/'meta/info.json').read_text())
assert info['fps']==15, 'RTC must preserve training 15Hz time scale'
assert info['features']['action']['shape']==[16]
assert info['features']['observation.state']['shape']==[16]
model_cameras={key for key in config['input_features'] if key.startswith('observation.images.')}
data_cameras={key for key in info['features'] if key.startswith('observation.images.')}
assert len(model_cameras)==4 and model_cameras==data_cameras
task_contract=resolve_task_contract(args, read_training_tasks(args.dataset_root))
print(json.dumps({'event':'rtc_preflight_ok','checkpoint':str(args.policy_path),
                  'task_contract':task_contract,
                  'dataset_root':str(args.dataset_root),
                  'guided':True,'frequency':15,'mask_mode':args.rtc_mask_mode,
                  'num_inference_steps':args.num_inference_steps,'hardware_commands_sent':False}))
PY
((PRECHECK)) && exit 0
[[ "${R1A7_REAL_INFERENCE:-}" == YES && "$CONFIRMED" == "$CONFIRMATION" ]] || {
  printf 'Real motion requires R1A7_REAL_INFERENCE=YES and exact confirmation.\n' >&2; exit 2;
}
[[ -t 0 ]] || { printf 'Interactive terminal required.\n' >&2; exit 2; }
if pgrep -f '[/]teleop_hand_and_arm.py|[/]r1a7_real_command_bridge.py|unitree_lerobot.r1a7_(live_real|live_dagger|rtc_live)' >/dev/null; then
  printf 'Another hardware publisher is running. Stop it before RTC.\n' >&2; exit 2
fi
"$PI_PYTHON" - <<'PY'
import socket
for port in (55580,55582):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',port))
PY
BRIDGE_PID=""
MODEL_PID=""
cleanup() {
  if [[ -n "$MODEL_PID" ]]; then
    kill -TERM "$MODEL_PID" 2>/dev/null || true
    wait "$MODEL_PID" 2>/dev/null || true
  fi
  if [[ -n "$BRIDGE_PID" ]]; then
    kill -TERM "$BRIDGE_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
PYTHONPATH="$ROOT_DIR/vendor/unitree-sdk2-python:$ROOT_DIR/data/unitree-lerobot:$ROOT_DIR/robot/xr-teleoperate:$ROOT_DIR/robot/xr-teleoperate/teleop/televuer/src" \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$TV_PYTHON" "$ROOT_DIR/robot/xr-teleoperate/teleop/tools/r1a7_real_command_bridge.py" \
  --xr-teleoperate-root "$ROOT_DIR/robot/xr-teleoperate" --network-interface "$NETWORK" \
  --bind tcp://127.0.0.1:55582 --command-timeout .5 --dex1-profile matched-teleop \
  --confirm-real-robot "$CONFIRMATION" &
BRIDGE_PID=$!
"$PI_PYTHON" - "$BRIDGE_PID" <<'PY'
import os,socket,sys,time
deadline=time.monotonic()+30
while time.monotonic()<deadline:
    os.kill(int(sys.argv[1]),0)
    try:
        with socket.create_connection(('127.0.0.1',55582),timeout=.2):
            break
    except OSError:
        time.sleep(.2)
else:
    raise SystemExit('RTC hardware bridge did not become ready')
PY
"$PI_PYTHON" -m unitree_lerobot.r1a7_rtc_language "${ARGS[@]}" <&0 &
MODEL_PID=$!
wait "$MODEL_PID"
