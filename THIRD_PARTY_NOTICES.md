# Third-party sources and notices

The following principal components were imported into the monorepo. Their original
license files have been retained. This list is a source inventory, not a replacement
for the license terms or an exhaustive review of every transitive dependency.

| Component | Upstream | License file |
| --- | --- | --- |
| Unitree XR teleoperation | https://github.com/unitreerobotics/xr_teleoperate | `robot/xr-teleoperate/LICENSE` (Apache-2.0 notice) |
| Unitree TeleImager | https://github.com/unitreerobotics/teleimager | `robot/xr-teleoperate/teleop/teleimager/LICENSE` (Apache-2.0 notice) |
| Unitree TeleVuer | https://github.com/unitreerobotics/televuer | `robot/xr-teleoperate/teleop/televuer/LICENSE` (MIT) |
| Dex retargeting | https://github.com/silencht/dex-retargeting | `robot/xr-teleoperate/teleop/robot_control/dex-retargeting/LICENSE` and `license_legacy/LICENSE` |
| Dex1 service | https://github.com/unitreerobotics/dex1_1_service | `robot/dex1-service/LICENSE` (BSD-3-Clause) |
| Unitree LeRobot wrapper | https://github.com/unitreerobotics/unitree_lerobot | `data/unitree-lerobot/LICENSE` (Apache-2.0 notice) |
| Hugging Face LeRobot | https://github.com/huggingface/lerobot | `data/unitree-lerobot/unitree_lerobot/lerobot/LICENSE` (Apache-2.0) |
| Unitree C++ SDK | https://github.com/unitreerobotics/unitree_sdk2 | `vendor/unitree-sdk2/LICENSE` (BSD-3-Clause) |
| Unitree Python SDK | https://github.com/unitreerobotics/unitree_sdk2_python | `vendor/unitree-sdk2-python/LICENSE` (BSD-3-Clause) |

Additional notices for bundled Cyclone DDS, iceoryx, and RapidJSON components
remain in `vendor/unitree-sdk2/licenses/` and the associated third-party directories.
Robot mesh files and prebuilt libraries are excluded from this release pending
separate redistribution review; obtain compatible artifacts from official upstreams.
Distributed integration source files carry a release modification notice. Original
upstream notices and component license files are retained.

`versions.lock` records the import baselines. Local modifications do not transfer
upstream authorship to the project, and the root Apache-2.0 license does not override
the licenses of imported components. Model weights and datasets are separate
artifacts and are not distributed in this candidate.
