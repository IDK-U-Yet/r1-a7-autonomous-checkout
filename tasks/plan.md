# R1-A7 human-intervention DAgger implementation

Approved scope: model rollout -> explicit PICO takeover -> human completion;
one hardware bridge, continuous four-camera recording, expert-only export.
The user approved this scope in chat on 2026-09-30. No live robot actuation by
the implementing agent. Existing data and unrelated working-tree changes stay intact.

## Capability map and build order
1. Ownership: MODEL -> HOLD -> HUMAN, epoch checked at the actuator boundary.
2. PICO: input-only standby, relative FK alignment, preserved gripper goal,
   measured-q IK seed; stale XR holds, never automatically returns to MODEL.
3. Recording: bounded asynchronous four-camera writer, source-trial and
   command timestamps; failures stop the run, incomplete trials not exported.
4. Runtime: asynchronous model worker, one scanner listener, explicit takeover
   and completion; recorder continues during HUMAN and HOLD.
5. Expert export: only contiguous HUMAN/task/epoch segments; preserve full
   parent-trial identity for later split grouping. No model labels in expert data.

## Contracts
- 16D: 14 arm joints + two Dex1 motor position targets; waist held outside model.
- Control 30 Hz for human, recording 15 Hz; camera freshness/skew gates reused.
- PICO right A requests takeover/alignment; right B holds. Re-press A realigns
  after XR interruption. Keyboard S completes, Q rejects; explicit R starts
  a new trial only after HOLD. No automatic reset while the operator is in the workspace.
- Continuous-cycle extension approved in chat on 2026-10-01: X/S saves only
  after this trial's accepted scan and operator-confirmed placement; Y/Q deletes
  only the current recorder's UUID trial after writer closure (updated by user).
  Neither starts motion. P/T holds. R is accepted only from HOLD or between
  trials, abandons an unfinished trial, invalidates old actions, returns to the
  start pose, then creates a fresh trial and scanner state. No automatic resume.
- The dataset-root folder name automatically groups UUID trial directories below
  the raw root; collection-name remains an optional backward-compatible override.
  Manifest records actual dataset/checkpoint paths as well as the human batch id.
- Bridge between-trial mode ignores takeover/buttons and preserves grip.
  Finalization runs off-thread while the coordinator maintains HOLD heartbeats.
  A bounded 12s grace is a fallback, not a replacement for live heartbeat.
  Hardware feedback checks remain active; storage failure prevents restart.
- Scanner callback shares a trial-switch lock; paused/old scan events cannot
  transition the next trial. Policy generation includes a monotonic cycle index.
- Completed is an operator assertion, not proof of successful task execution.
  Right inner grip is also an explicit R request in HOLD/between trials, with
  fresh release/press edges only; active/held/finalization presses cannot replay.
  Export also requires accepted scanner event for successful checkout trials.
- No changes to legacy data; exports are new directories. Raw policy rollout
  is audit-only and is never directly sent to behavior-cloning training.

## Verification and boundaries
Use pytest on pure ownership/alignment/writer/export contracts and existing
bridge tests; shell syntax checks and help/preflight do not move hardware.
Mocked tests do not establish safe handover latency on real hardware. The
operator must first test takeover with an empty gripper in an open workspace.
No remote upload, training launch, destructive cleanup, or live actuation.
Style: existing Python/numpy/ZMQ conventions; lazy heavy PICO/IK imports.
Risks: IK discontinuity, trigger mismatch, stale chunk races, camera gaps,
filesystem stalls, duplicate DDS publishers; all explicitly gated or recorded.

## Extension implementation order and verification
### Approved takeover mapping extension (2026-10-02)
User confirmed fading the initial robot/XR offset to restore ordinary teleop
mapping, not fading a position toward the origin. Translation uses current XR
position plus (1-weight) times initial robot-minus-XR position. Rotation uses a
shortest-path SO(3) anchor blend followed by current XR relative rotation.
Default minimum duration 3 seconds, optional --takeover-blend-seconds; 0 keeps
fixed-relative mapping. Quintic smoothstep duration includes 1.875 derivative
factor to cap automatic offset removal at 0.10 m/s and 0.6 rad/s. These caps do
not limit deliberate XR motion. Advance clock only during valid HUMAN updates,
with elapsed step capped at 1/30s to prevent stall recovery jumps. A starts at
zero weight with no activation-edge command; HOLD discards transition, A reanchors.
Keep IK/freshness/joint/gripper gates. Record blend state; blended automatic
motion is audit-only (expert_action_valid false until fully ordinary mapping).
Build order: pure mapping/clock tests -> bridge/CLI/metadata -> expert filtering
and mock regressions -> documentation and independent review. No hardware tests.

1. Bridge lifecycle: between-trial HOLD, finish acknowledgement, epoch-safe restart.
2. Runtime lifecycle: P/R, X/Y finish gate, unique generations and recorder switch.
3. Batch naming: shell/preflight validation, recording/export metadata.
4. Mock two-cycle and malformed-name tests, legacy regressions, operator runbook.
Test: PYTHONPATH=data/unitree-lerobot:runtime/lerobot-a100-262d2fd/src:robot/xr-teleoperate runtime/envs/pi05-262d2fd/bin/python -m pytest -q data/unitree-lerobot/test/test_r1a7_live_dagger.py data/unitree-lerobot/test/test_r1a7_dagger.py
Keep existing Python/JSON conventions; e.g. recorder.event("trial_saved", completed=True).
Always preserve old datasets; never actuate hardware or launch training here.
