# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import json
import subprocess


def _ipv4_interfaces():
    result = subprocess.run(
        ["ip", "-j", "-4", "addr", "show"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def resolve_dds_interface(requested, expected_ipv4="192.168.123.222"):
    """Resolve a renamed USB NIC by its configured robot-subnet address."""
    interfaces = _ipv4_interfaces()
    addresses = {
        item["ifname"]: {
            entry["local"]
            for entry in item.get("addr_info", [])
            if entry.get("family") == "inet"
        }
        for item in interfaces
    }
    if requested in addresses and expected_ipv4 in addresses[requested]:
        return requested, False

    candidates = [
        name for name, values in addresses.items() if expected_ipv4 in values
    ]
    if len(candidates) == 1:
        return candidates[0], candidates[0] != requested
    if not candidates:
        raise RuntimeError(
            f"no network interface has required DDS address {expected_ipv4}; "
            f"requested interface was {requested!r}"
        )
    raise RuntimeError(
        f"multiple interfaces have DDS address {expected_ipv4}: {candidates}"
    )
