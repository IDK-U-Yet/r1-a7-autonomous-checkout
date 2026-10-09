# 现场硬件映射模板

复制本模板到 Git 忽略的现场配置目录后填写。不要在仓库中提交令牌、证书、序列号与不必要的网络拓扑。

```text
DDS_INTERFACE=<network-interface>
ROBOT_HOST=<robot-ip>
OPERATOR_HOST=<operator-ip>
PICO_HOST=<pico-ip>
DEX1_RIGHT_MOTOR_ID=0
DEX1_LEFT_MOTOR_ID=1
SCANNER_DEVICE=/dev/<stable-scanner-link>
GLOBAL_CAMERA_ID=<device-id>
LEFT_WRIST_CAMERA_ID=<device-id>
RIGHT_WRIST_CAMERA_ID=<device-id>
```
