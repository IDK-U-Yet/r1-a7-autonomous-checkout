# Human-intervention DAgger

- [x] Ownership + relative pose/gripper alignment (epoch rejection tests).
- [x] Optional PICO control inside sole hardware bridge (legacy regression tests).
- [x] Bounded raw recording + human-only export (gap/task/epoch/error tests).
- [x] Dedicated asynchronous runtime + launcher (syntax/help and mock checks).
- [x] Single-model code review, fix findings, operator runbook.
- [ ] Operator-only empty-gripper hardware gate (not performed by Codex).

## Continuous cycles (approved 2026-10-01)
- [x] Epoch-safe bridge wait/restart; no A motion between trials.
- [x] X/Y finalize without exit; P pause and R restart; old scanner/chunks isolated.
- [x] Validated collection-name folder and actual source metadata.
- [x] Two-cycle software regressions and updated launch instructions.
- [x] Y/Q deletes only the current closed trial; older and R-abandoned audits retained.
- [x] Right inner grip invokes R only from fresh HOLD/between-trial press; no held repeat.

## Takeover offset fade (approved 2026-10-02)
- [x] Smooth position/SO(3) offset fade, minimum duration and stall-safe clock; 0 restores legacy mode.
- [x] Bridge/launcher configuration, HOLD/reanchor boundaries and raw blend metadata.
- [x] Exclude automatic fade frames from expert export, preserving original audit.
- [x] Endpoint/rate/clock/export/CLI regressions and independent single-model review.
- [ ] Operator-only empty-gripper test before using fade for held-product recovery.
