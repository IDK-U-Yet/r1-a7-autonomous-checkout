# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
import torch

from lerobot.configs import RTCAttentionSchedule
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from unitree_lerobot.r1a7_rtc_guidance import R1RTCProcessor


def test_guidance_contains_denoiser_jacobian_under_no_grad():
    config = RTCConfig(prefix_attention_schedule=RTCAttentionSchedule.ONES,
                       max_guidance_weight=1, execution_horizon=2)
    processor = R1RTCProcessor(config)
    x = torch.ones(1, 2, 16)
    prefix = torch.zeros_like(x)
    # v=0.5*x, x1=x-0.5*v=0.75*x. Guidance correction is
    # (prefix-x1) * d(x1)/dx = -0.75*0.75, not just -0.75.
    with torch.no_grad():
        result = processor.denoise_step(x, prefix, 1, .5, lambda z: .5*z)
    torch.testing.assert_close(result, torch.full_like(x, 1.0625))
    assert not result.requires_grad


def test_no_prefix_keeps_original_velocity():
    processor = R1RTCProcessor(RTCConfig())
    x = torch.ones(1, 50, 32)
    with torch.no_grad():
        result = processor.denoise_step(x, None, 0, .5, lambda z: z*2)
    torch.testing.assert_close(result, x*2)
