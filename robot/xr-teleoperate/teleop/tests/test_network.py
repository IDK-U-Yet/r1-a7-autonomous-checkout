# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
# Modified for the R1-A7 public source release: workstation addresses use examples.
# Original component notices and license remain applicable; see versions.lock.
import unittest
from unittest.mock import patch

from teleop.utils.network import resolve_dds_interface


def interface(name, *addresses):
    return {
        "ifname": name,
        "addr_info": [
            {"family": "inet", "local": address} for address in addresses
        ],
    }


class ResolveDDSInterfaceTest(unittest.TestCase):
    @patch("teleop.utils.network._ipv4_interfaces")
    def test_keeps_requested_interface_when_address_matches(self, get_interfaces):
        get_interfaces.return_value = [interface("enx-old", "192.168.123.222")]
        self.assertEqual(
            resolve_dds_interface("enx-old"), ("enx-old", False)
        )

    @patch("teleop.utils.network._ipv4_interfaces")
    def test_recovers_usb_interface_rename_by_address(self, get_interfaces):
        get_interfaces.return_value = [
            interface("wlan0", "192.168.1.100"),
            interface("enx-new", "192.168.123.222"),
        ]
        self.assertEqual(
            resolve_dds_interface("enx-old"), ("enx-new", True)
        )

    @patch("teleop.utils.network._ipv4_interfaces")
    def test_rejects_missing_robot_address(self, get_interfaces):
        get_interfaces.return_value = [interface("wlan0", "192.168.1.100")]
        with self.assertRaises(RuntimeError):
            resolve_dds_interface("enx-old")


if __name__ == "__main__":
    unittest.main()
