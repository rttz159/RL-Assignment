import numpy as np
from DigitalTwinModel import DigitalTwinModel


class HybridLookaheadPolicyWrapper:
    """Lookahead planner with RL Model and hybrid emergency override for glucose extremes."""

    SAFE_LOW = 70.0
    SAFE_HIGH = 180.0
    CRITICAL_LOW = 50.0
    CRITICAL_HIGH = 250.0

    def __init__(
        self,
        model,
        digital_twin: DigitalTwinModel,
        horizon=5,
        candidates=5,
        target_glucose=100.0,
    ):
        self.model = model
        self.twin = digital_twin
        self.horizon = int(horizon)
        self.candidates = int(candidates)
        self.target = float(target_glucose)

    def _score_traj(self, traj_obs):
        G = traj_obs[:, 0]
        dev_penalty = -np.mean(np.abs(G - self.target))
        swing_penalty = -np.mean(np.maximum(0.0, np.abs(np.diff(G)) - 10.0))
        safety = -np.sum(np.maximum(0, self.SAFE_LOW - G) * 50) - np.sum(
            np.maximum(0, G - self.CRITICAL_HIGH) * 20
        )
        return 1000 + dev_penalty + 0.5 * swing_penalty + safety

    def predict(self, obs, env_state, deterministic=False):
        glucose = obs[0]

        if glucose < self.SAFE_LOW:
            deficit = self.SAFE_LOW - glucose
            meal_fraction = np.clip(deficit / 50.0, 0.3, 0.7)
            return (
                np.array([meal_fraction, 0.0, 0.0, 0.0], dtype=np.float32),
                "hypoglycemia override",
            )

        if glucose > self.SAFE_HIGH:
            excess = glucose - self.SAFE_HIGH
            intensity = np.clip(excess / 100.0, 0.3, 0.7)
            duration = 3 / 12.0
            return (
                np.array([0.0, 1.0, intensity, duration], dtype=np.float32),
                "hyperglycemia override",
            )

        sampled_actions = [
            self.model.predict(obs, deterministic=deterministic)[0]
            for _ in range(self.candidates)
        ]
        best_action, best_score = None, -np.inf

        for first_action in sampled_actions:
            traj_obs, _ = self.twin.rollout_from_env_state(
                env_state,
                first_action,
                self.model,
                horizon=self.horizon,
                deterministic=deterministic,
            )
            score = self._score_traj(traj_obs)
            if score > best_score:
                best_score = score
                best_action = first_action

        return best_action, "RL Model lookahead"
