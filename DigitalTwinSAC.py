import numpy as np
from scipy.integrate import solve_ivp
from GlucoseDynamicsSimulator import GlucoseDynamicsSimulator


class DigitalTwinSAC:
    """Deterministic digital twin of CustomGlucoseDynamicsEnvBox for planning."""

    def __init__(self, weight=70.0, Gb=90.0, Ib=15.0, dt=5):
        self.sim = GlucoseDynamicsSimulator(weight=weight, Gb=Gb, Ib=Ib)
        self.dt = dt
        self.latent_state = None
        self.current_exercise_session = {
            "active": False,
            "remaining_steps": 0,
            "intensity": 0.0,
        }
        self.time_since_midnight = 0.0
        self.previous_glucose_level = None

    @staticmethod
    def _continuous_to_discrete(action):
        action = np.clip(np.array(action, dtype=np.float32), 0.0, 1.0)
        meal = int(np.round(action[0] * 3))
        mode = int(np.round(action[1]))
        intensity = int(np.round(action[2] * 3))
        duration = int(np.round(action[3] * 12))
        return [meal, mode, intensity, duration]

    @staticmethod
    def _map_meal_category_to_glucose_rate(category):
        return [0.0, 0.45, 0.65, 0.8][category]

    @staticmethod
    def _map_exercise_level_to_intensity(level):
        return [0.0, 0.3, 0.6, 0.9][level]

    def _build_observation_from_latent(self):
        G = float(self.latent_state[0])
        I_plasma = float(self.latent_state[7])
        time_since_last_meal = float(self.latent_state[5])
        time_since_last_exercise = float(self.latent_state[6])
        L = float(self.latent_state[4])
        t = float(self.time_since_midnight)
        prev_glucose = float(
            G if self.previous_glucose_level is None else self.previous_glucose_level
        )
        obs = np.array(
            [
                G,
                prev_glucose,
                I_plasma,
                time_since_last_meal,
                1.0 if 6 * 60 <= t < 9 * 60 else 0.0,
                1.0 if 12 * 60 <= t < 14 * 60 else 0.0,
                1.0 if 18 * 60 <= t < 20 * 60 else 0.0,
                time_since_last_exercise,
                L,
                t,
            ],
            dtype=np.float32,
        )
        return obs

    def sync_from_env_state(self, env_state):
        self.latent_state = env_state["simulator_state"].copy()
        self.time_since_midnight = env_state["time_since_midnight"]
        self.previous_glucose_level = env_state["previous_glucose_level"]
        self.current_exercise_session = env_state["current_exercise_session"].copy()

    def simulate_step(self, action, env_state):
        self.sync_from_env_state(env_state)
        return self._advance_one_step(action)

    def _advance_one_step(self, action):
        meal_cat, exercise_mode, exercise_intensity, exercise_duration = (
            self._continuous_to_discrete(action)
        )
        Rameal_current = self._map_meal_category_to_glucose_rate(meal_cat)

        if exercise_mode == 1 and not self.current_exercise_session["active"]:
            self.current_exercise_session["active"] = True
            self.current_exercise_session["remaining_steps"] = exercise_duration
            self.current_exercise_session["intensity"] = (
                self._map_exercise_level_to_intensity(exercise_intensity)
            )
        elif exercise_mode != 1:
            self.current_exercise_session = {
                "active": False,
                "remaining_steps": 0,
                "intensity": 0.0,
            }

        if self.current_exercise_session["active"]:
            self.current_exercise_session["remaining_steps"] -= 1
            if self.current_exercise_session["remaining_steps"] <= 0:
                self.current_exercise_session = {
                    "active": False,
                    "remaining_steps": 0,
                    "intensity": 0.0,
                }

        E_current = float(self.current_exercise_session["intensity"])
        ode_func = lambda t, y: self.sim.odes(
            y, (Rameal_current, E_current, meal_cat > 0, exercise_mode == 1)
        )
        sol = solve_ivp(
            ode_func,
            t_span=(0, self.dt),
            y0=self.latent_state,
            method="RK45",
            max_step=0.5,
        )
        self.latent_state = sol.y[:, -1]
        self.time_since_midnight += self.dt
        self.previous_glucose_level = float(self.latent_state[0])
        return self._build_observation_from_latent()

    def rollout_from_env_state(
        self, env_state, first_action, policy, horizon=5, deterministic=False
    ):
        backup = self._snapshot()
        self.sync_from_env_state(env_state)
        traj_obs, actions = [], []
        a = np.array(first_action, dtype=np.float32)
        for _ in range(horizon):
            next_obs = self._advance_one_step(a)
            traj_obs.append(next_obs.copy())
            a, _ = policy.predict(next_obs, deterministic=deterministic)
            actions.append(a.copy())
        self._restore(backup)
        return np.array(traj_obs), np.array(actions)

    def _snapshot(self):
        return {
            "latent_state": (
                None if self.latent_state is None else self.latent_state.copy()
            ),
            "time_since_midnight": float(self.time_since_midnight),
            "previous_glucose_level": (
                None
                if self.previous_glucose_level is None
                else float(self.previous_glucose_level)
            ),
            "current_exercise_session": self.current_exercise_session.copy(),
        }

    def _restore(self, snap):
        self.latent_state = (
            None if snap["latent_state"] is None else snap["latent_state"].copy()
        )
        self.time_since_midnight = float(snap["time_since_midnight"])
        self.previous_glucose_level = (
            None
            if snap["previous_glucose_level"] is None
            else float(snap["previous_glucose_level"])
        )
        self.current_exercise_session = snap["current_exercise_session"].copy()
