# R1-A7 public snapshot: integration/release edits; upstream baselines in versions.lock.
"""Independent guided-RTC adapter; prefixes stay in checkpoint model space."""

from contextlib import nullcontext
import time

import numpy as np

from unitree_lerobot.r1a7_contract import CAMERA_KEYS, ROBOT_TYPE, ACTION_LOWER_LIMITS, ACTION_UPPER_LIMITS, JOINT_NAMES
from unitree_lerobot.r1a7_live_shadow import PI05Session


class RTCSession(PI05Session):
    def __init__(self, *, execution_horizon=10, max_guidance_weight=10.0,
                 mask_mode="configured", num_inference_steps=10, **kwargs):
        if type(num_inference_steps) is not int or num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer")
        self.num_inference_steps = num_inference_steps
        if mask_mode not in ("configured", "overlap"):
            raise ValueError("unknown RTC mask mode")
        self.mask_mode = mask_mode
        self.mask_endpoint = execution_horizon
        super().__init__(**kwargs)
        from lerobot.configs import RTCAttentionSchedule
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        if self.policy.config.type != "pi05":
            raise ValueError("R1 RTC requires a pi05 checkpoint")
        if not 0 < execution_horizon < self.chunk_size:
            raise ValueError("execution_horizon must be within the model chunk")
        self.policy.config.rtc_config = RTCConfig(
            enabled=True, mode="guided", execution_horizon=execution_horizon,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
            max_guidance_weight=max_guidance_weight,
        )
        self.policy.init_rtc_processor()
        from unitree_lerobot.r1a7_rtc_guidance import R1RTCProcessor
        processor = R1RTCProcessor(self.policy.config.rtc_config)
        self.policy.rtc_processor = processor
        self.policy.model.rtc_processor = processor
        self.policy.requires_grad_(False)
        # Use the checkpoint's action statistics, not observation-state statistics.
        from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
        normalizers = [step for step in self.postprocessor.steps
                       if isinstance(step, UnnormalizerProcessorStep)]
        if len(normalizers) != 1:
            raise ValueError("RTC requires exactly one checkpoint action unnormalizer")
        bounds = self.torch.as_tensor(
            np.stack([ACTION_LOWER_LIMITS, ACTION_UPPER_LIMITS]),
            dtype=self.torch.float32, device=self.device,
        )
        with self.torch.no_grad():
            self._normalized_bounds = normalizers[0]._normalize_action(
                bounds, inverse=False).detach().float().cpu().numpy()

    def project_targets(self, actions):
        """Project all targets before caching the prefix; never expand limits.

        This prevents normal hard-limit saturation from invalidating the whole
        plan. It does not certify collision freedom or task success.
        """
        source = np.asarray(actions, dtype=np.float32)
        if source.ndim != 2 or source.shape[1] != 16 or not np.isfinite(source).all():
            raise ValueError("RTC projection requires finite (N,16) actions")
        result = np.clip(source, self._normalized_bounds[0], self._normalized_bounds[1])
        delta = np.abs(result-source)
        self.last_projection_summary = {
            "projected_frames": int(np.count_nonzero(np.any(delta > 0, axis=1))),
            "joint_counts": {name: int(np.count_nonzero(delta[:, i]))
                             for i, name in enumerate(JOINT_NAMES) if np.any(delta[:, i])},
            "max_delta_normalized": float(delta.max()) if len(delta) else 0.,
        }
        return result

    def predict_normalized(self, state, images, task, *, prefix=None, delay_steps=0):
        """Return normalized (T,16); do not roundtrip prefix through hardware units."""
        torch = self.torch
        if delay_steps < 0 or delay_steps >= self.chunk_size:
            raise ValueError("delay_steps must be within the model chunk")
        observation = {
            "observation.state": torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0),
            "task": task, "robot_type": ROBOT_TYPE,
        }
        for key in CAMERA_KEYS:
            observation[key] = torch.as_tensor(
                images[key], dtype=torch.float32, device=self.device
            ).unsqueeze(0) / 255.0
        previous = None
        if prefix is not None and len(prefix):
            array = np.asarray(prefix, dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != 16 or not np.isfinite(array).all():
                raise ValueError("prefix must be finite normalized (T,16)")
            if len(array) > self.chunk_size or delay_steps > len(array):
                raise ValueError("prefix does not cover the committed delay")
            previous = torch.as_tensor(array.copy(), device=self.device).unsqueeze(0)
        amp = (torch.autocast(device_type="cuda")
               if self.device.type == "cuda" and self.use_amp else nullcontext())
        # no_grad permits RTC's inner enable_grad; inference_mode would forbid it.
        with torch.inference_mode(False), torch.no_grad(), amp:
            batch = self.preprocessor(observation)
            endpoint = (len(prefix) if self.mask_mode == "overlap" else self.mask_endpoint) if previous is not None else None
            self.last_mask_endpoint = min(endpoint, len(prefix)) if endpoint is not None else None
            # Synchronize only the inference worker, never the dispatch thread.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            model_started = time.monotonic_ns()
            actions = self.policy.predict_action_chunk(
                batch, inference_delay=delay_steps, prev_chunk_left_over=previous,
                execution_horizon=endpoint,
                num_steps=self.num_inference_steps,
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.last_model_latency_ms = (time.monotonic_ns() - model_started) / 1e6
        result = actions.squeeze(0).detach().float().cpu().numpy()
        if result.shape != (self.chunk_size, 16) or not np.isfinite(result).all():
            raise ValueError("RTC returned invalid normalized actions")
        return self.project_targets(result)

    def to_hardware_action(self, normalized):
        torch = self.torch
        with torch.inference_mode(False), torch.no_grad():
            action = torch.as_tensor(
                normalized, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            return self.postprocessor(action).squeeze(0).detach().double().cpu().numpy()
