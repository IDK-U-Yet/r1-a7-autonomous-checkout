# Public source release scope

The maintainer authorized a new public repository on 2026-10-09, separate from the
private development repository. This is a source snapshot with a fresh history,
not a mirror of private history and not a turnkey end-to-end deployment.

## Included and checked

- Project-owned original code is Apache-2.0; imported components retain their
  licenses and notices. Upstream origins and import baselines are in `versions.lock`.
- Collection monitoring, dataset playback/review/annotation, conversion, training,
  guarded inference, DAgger takeover, and RTC source are included.
- Integration source files carry public-snapshot modification notices.
- Selected workstation paths and deployment addresses were replaced with examples.
- Initial source checks covered 52 shell scripts and preserved 15 imported license
  files; 16 network/scan-event unit tests passed without hardware commands.
- Gitleaks 8.30.1 found one truncated private-key example in upstream DDS
  documentation. It was manually reviewed, is not a complete key, and was retained.
  Scanner checks are not proof of the absence of all private content.

## Deliberately excluded

- The private checkout application and its dedicated launcher.
- Datasets, checkpoints, databases, runtime logs/environments, private keys,
  interview documents, original Git history, and imported CI workflows.
- Robot mesh files and precompiled libraries pending separate redistribution
  review. Fetch matching artifacts from official upstreams before building/running
  affected tools; obey their respective license terms.

## Known limitations

- The barcode bridge still depends on `services.scanner_service.hid` and the
  SQLite catalog schema from the excluded private checkout application. The
  maintainer chose to publish with this dependency documented, not to implement
  a replacement. The scanner-driven workflow does not run independently.
- A clean-machine installation and full hardware validation have not been performed
  for this public snapshot. Passing source/unit checks is not hardware validation.
- Legacy workflow documents describe experiment-specific configurations and the
  full private system. References to excluded components are not runnable examples.
- No model weights or datasets are bundled; supply compatible four-camera/16D
  artifacts and their processors/statistics yourself.

The original development repository remains unchanged. Subsequent updates require
deliberate export and review. Personal application/interview documents stay local.
