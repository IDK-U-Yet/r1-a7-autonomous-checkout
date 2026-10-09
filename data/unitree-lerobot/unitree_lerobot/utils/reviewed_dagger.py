# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Explicit, provenance-preserving approval of reviewed DAgger targets."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


MANIFEST = "reviewed_training_export.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_reviewed_export(root, episode_paths):
    """Checks consistency, not cryptographic authentication of the reviewer."""
    root = Path(root).resolve()
    path = root / MANIFEST
    if not path.is_file():
        return set()
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("export_type") != "human_reviewed_dagger":
        raise ValueError("Invalid reviewed DAgger manifest")
    entries = manifest.get("episodes", [])
    expected = {str(Path(p).resolve().relative_to(root) / "data.json") for p in episode_paths}
    if len(entries) != len(expected) or {e["path"] for e in entries} != expected:
        raise ValueError("Reviewed manifest episode set mismatch")
    for entry in entries:
        source = (root / entry["path"]).resolve()
        if not source.is_relative_to(root) or digest(source) != entry["sha256"]:
            raise ValueError("Reviewed episode checksum mismatch")
        approval = entry.get("approval", {})
        if approval.get("review_status") != "approved" or approval.get("revision", 0) < 1:
            raise ValueError("Reviewed episode lacks approval")
        payload = json.loads(source.read_text())
        if not payload.get("info", {}).get("dagger_reviewed_export") or payload["info"].get("dagger_audit_only"):
            raise ValueError("Reviewed export info mismatch")
        samples = payload.get("data", [])
        if len(samples) != entry["frames"] or not samples:
            raise ValueError("Reviewed frame count mismatch")
        for sample in samples:
            meta = sample.get("metadata", {})
            if meta.get("control_source") not in {"human", "hold"} or meta.get("sample_valid") is not True:
                raise ValueError("Reviewed export requires valid HUMAN/HOLD targets; POLICY is forbidden")
            if meta["control_source"] == "hold" and manifest.get("include_reviewed_hold") is not True:
                raise ValueError("HOLD requires explicit reviewed authorization")
            if meta.get("expert_action_valid") is not True and manifest.get("include_reviewed_transition") is not True:
                raise ValueError("Transition requires explicit reviewed authorization")
    return expected


def export(source, destination, annotations, include_hold=False, include_transition=False):
    from dataset_review.catalog import Catalog
    from dataset_review.annotations import AnnotationStore

    source, destination = Path(source).resolve(), Path(destination).resolve()
    dataset = Catalog([source]).datasets[0]
    store = AnnotationStore(Path(annotations))
    approved, excluded = [], []
    # Validate the whole selection before creating any output.
    for trajectory in dataset.trajectories("full"):
        approval = store.get(trajectory)
        if approval["review_status"] != "approved":
            excluded.append({"episode": trajectory.name, "status": approval["review_status"]})
            continue
        if any(i["exclude_from_training"] for i in approval["quality_intervals"]):
            raise ValueError(f"Unapplied training exclusion in {trajectory.name}")
        if len(trajectory.segments) != 1:
            raise ValueError("Expected one raw episode per reviewed trajectory")
        raw = trajectory.segments[0].raw_dir
        payload = json.loads((raw / "data.json").read_text())
        for sample in payload["data"]:
            meta = sample.get("metadata", {})
            if meta.get("control_source") not in {"human", "hold"} or meta.get("sample_valid") is not True:
                raise ValueError(f"Invalid/non-expert target in {trajectory.name}")
            if meta["control_source"] == "hold" and not include_hold:
                raise ValueError("Pass --include-reviewed-hold after manual approval")
            if meta.get("expert_action_valid") is not True and not include_transition:
                raise ValueError("Pass --include-reviewed-transition after manual approval")
        approved.append((raw, payload, approval))
    if not approved or not dataset.source_unchanged():
        raise ValueError("No approved episodes or source metadata changed")
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "export_type": "human_reviewed_dagger",
                "source_root": str(source), "source_fingerprint": dataset.fingerprint,
                "include_reviewed_hold": include_hold, "include_reviewed_transition": include_transition,
                "boundary_policy": "split_training_sequences_at_continuity_changes",
                "excluded": excluded, "episodes": []}
    for raw, payload, approval in approved:
        out = destination / raw.name
        out.mkdir()
        # Independent media copies: later edits/deletions of the audit cannot alter this export.
        for sample in payload["data"]:
            for value in sample.get("colors", {}).values():
                image = (raw / value).resolve()
                if not image.is_relative_to(raw.resolve()):
                    raise ValueError("Image path outside source episode")
                target = out / value
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(image, target)
        payload["info"].update(dagger_audit_only=False, dagger_reviewed_export=True,
                               dagger_expert_export=False, training_enabled=True,
                               quality_status="human_reviewed_approved")
        path = out / "data.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False))
        manifest["episodes"].append({"path": str(path.relative_to(destination)),
                                     "sha256": digest(path), "frames": len(payload["data"]),
                                     "source_episode": str(raw), "approval": approval})
    for name in ("takeover_review_mapping.json", "provenance"):
        origin = source / name
        if origin.is_dir():
            shutil.copytree(origin, destination / name)
        elif origin.is_file():
            shutil.copyfile(origin, destination / name)
    # Completion marker is written last. Partial exports cannot pass conversion.
    (destination / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--include-reviewed-hold", action="store_true")
    parser.add_argument("--include-reviewed-transition", action="store_true")
    args = parser.parse_args()
    manifest = export(args.source, args.destination, args.annotations,
                      args.include_reviewed_hold, args.include_reviewed_transition)
    print(json.dumps({"episodes": len(manifest["episodes"]),
                      "frames": sum(e["frames"] for e in manifest["episodes"]),
                      "destination": str(args.destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
