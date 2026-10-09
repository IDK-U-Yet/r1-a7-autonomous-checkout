#!/usr/bin/env bash
set -Eeuo pipefail

readonly LEFT_WRIST="/dev/v4l/by-id/usb-JR0002_JR0002_JR0002-video-index0"
readonly RIGHT_WRIST="/dev/v4l/by-id/usb-JR0001_JR0001_JR0001-video-index0"

readonly EXPOSURE="${R1A7_WRIST_EXPOSURE:-250}"
readonly GAIN="${R1A7_WRIST_GAIN:-64}"
readonly BRIGHTNESS="${R1A7_WRIST_BRIGHTNESS:-20}"
readonly SATURATION="${R1A7_WRIST_SATURATION:-64}"
readonly SCANNER_GAIN="${R1A7_SCANNER_GAIN:-0}"
readonly SCANNER_BRIGHTNESS="${R1A7_SCANNER_BRIGHTNESS:-0}"

readonly MODE="${1:-}"

if [[ "$MODE" != "fixed" && "$MODE" != "auto" ]]; then
  printf 'Usage: %s fixed|auto\n' "$0" >&2
  exit 2
fi

command -v v4l2-ctl >/dev/null || {
  printf 'Missing v4l2-ctl; install v4l-utils.\n' >&2
  exit 1
}

for device in "$LEFT_WRIST" "$RIGHT_WRIST"; do
  [[ -e "$device" ]] || {
    printf 'Missing wrist camera: %s\n' "$device" >&2
    exit 1
  }
  if [[ "$MODE" == "fixed" ]]; then
    v4l2-ctl -d "$device" \
      --set-ctrl="brightness=$BRIGHTNESS" \
      --set-ctrl="saturation=$SATURATION" \
      --set-ctrl=backlight_compensation=0 \
      --set-ctrl=auto_exposure=1 \
      --set-ctrl=exposure_dynamic_framerate=0 \
      --set-ctrl="exposure_time_absolute=$EXPOSURE" \
      --set-ctrl="gain=$GAIN" \
      --set-ctrl=white_balance_automatic=1
  else
    v4l2-ctl -d "$device" \
      --set-ctrl=brightness=0 \
      --set-ctrl=backlight_compensation=1 \
      --set-ctrl=auto_exposure=3 \
      --set-ctrl=exposure_dynamic_framerate=0 \
      --set-ctrl="gain=$GAIN" \
      --set-ctrl=white_balance_automatic=1
  fi
done

mapfile -t scanner_candidates < <(compgen -G '/dev/v4l/by-id/*AY275520029-video-index0')
if [[ ${#scanner_candidates[@]} -ne 1 ]]; then
  printf 'Expected exactly one scanner-area camera with serial AY275520029, found %d.\n' \
    "${#scanner_candidates[@]}" >&2
  exit 1
fi
readonly SCANNER_CAMERA="${scanner_candidates[0]}"

if [[ "$MODE" == "fixed" ]]; then
  v4l2-ctl -d "$SCANNER_CAMERA" \
    --set-ctrl="brightness=$SCANNER_BRIGHTNESS" \
    --set-ctrl=backlight_compensation=0 \
    --set-ctrl=auto_exposure=3 \
    --set-ctrl=exposure_dynamic_framerate=0 \
    --set-ctrl="gain=$SCANNER_GAIN" \
    --set-ctrl=white_balance_automatic=1
  printf 'Camera controls applied: wrists exposure=%s gain=%s brightness=%s saturation=%s; scanner auto-exposure gain=%s brightness=%s; automatic white balance enabled.\n' \
    "$EXPOSURE" "$GAIN" "$BRIGHTNESS" "$SATURATION" \
    "$SCANNER_GAIN" "$SCANNER_BRIGHTNESS"
else
  v4l2-ctl -d "$SCANNER_CAMERA" \
    --set-ctrl=brightness=0 \
    --set-ctrl=backlight_compensation=1 \
    --set-ctrl=auto_exposure=3 \
    --set-ctrl=exposure_dynamic_framerate=1 \
    --set-ctrl=gain=0 \
    --set-ctrl=white_balance_automatic=1
  printf 'Automatic wrist and scanner-area exposure/white balance restored for diagnostics.\n'
fi
