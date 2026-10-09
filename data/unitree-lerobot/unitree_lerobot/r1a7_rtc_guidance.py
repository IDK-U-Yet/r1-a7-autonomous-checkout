# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Isolated Jacobian-order correction for the pinned RTC implementation.

The pinned processor enables x_t gradients *after* evaluating the denoiser,
which drops its Jacobian from guidance. Keep vendor code untouched and make
the latent differentiable before invoking the original denoiser.
"""

from lerobot.policies.rtc.modeling_rtc import RTCProcessor


class R1RTCProcessor(RTCProcessor):
    def denoise_step(self, x_t, prev_chunk_left_over, inference_delay, time,
                     original_denoise_step_partial, execution_horizon=None):
        def differentiable_denoiser(latent):
            latent.requires_grad_(True)
            return original_denoise_step_partial(latent)

        return super().denoise_step(
            x_t, prev_chunk_left_over, inference_delay, time,
            differentiable_denoiser if prev_chunk_left_over is not None
            else original_denoise_step_partial,
            execution_horizon,
        ).detach()
