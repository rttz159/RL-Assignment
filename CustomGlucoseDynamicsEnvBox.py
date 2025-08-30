import numpy as np
import gymnasium as gym
from gymnasium import spaces
from GlucoseDynamicsSimulator import GlucoseDynamicsSimulator
from scipy.integrate import solve_ivp


class CustomGlucoseDynamicsEnvBox(gym.Env):
    def __init__(
        self,
        random_events=True,
        weight=70.0,  # Can be float (fixed) or tuple (range)
        Gb=90.0,  # mg/dL
        Ib=5.7,
    ):  # µU/mL

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array(
                [700.0, 700.0, 200.0, 1440.0, 1.0, 1.0, 1.0, 1440.0, 1.0, 1440.0]
            ),
            dtype=np.float32,
        )

        self.random_events = random_events

        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0]), high=np.array([1, 1, 1, 1]), dtype=np.float32
        )

        self.weight = weight
        self.Gb = Gb
        self.Ib = Ib

        self.dt = 5
        self.previous_glucose_level = None

        self.simulator = None
        self.simulator_state = None
        self.time_since_midnight = None
        self.current_exercise_session = {
            "active": False,
            "remaining_steps": 0,
            "intensity": 0.0,
        }

        self.transition_log = []

    def continuous_to_discrete(self, action):
        action = np.clip(action, 0.0, 1.0)
        meal = int(np.round(action[0] * 3))
        mode = int(np.round(action[1]))
        intensity = int(np.round(action[2] * 3))
        duration = int(np.round(action[3] * 12))
        return [meal, mode, intensity, duration]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.previous_observation = None

        weight = (
            np.random.uniform(*self.weight)
            if isinstance(self.weight, tuple)
            else self.weight
        )
        Gb = np.random.uniform(*self.Gb) if isinstance(self.Gb, tuple) else self.Gb
        Ib = np.random.uniform(*self.Ib) if isinstance(self.Ib, tuple) else self.Ib

        self.previous_glucose_level = Gb
        self.simulator = GlucoseDynamicsSimulator(weight=weight, Gb=Gb, Ib=Ib)
        self.simulator_state = self.simulator.initial_conditions()

        G0 = self.simulator_state[0]
        I_plasma0 = self.simulator_state[7]
        time_since_last_meal = self.simulator_state[5]
        time_since_last_exercise = self.simulator_state[6]

        self.time_since_midnight = 0.0
        is_breakfast_window = (
            1.0 if 6 * 60 <= self.time_since_midnight < 9 * 60 else 0.0
        )
        is_lunch_window = 1.0 if 12 * 60 <= self.time_since_midnight < 14 * 60 else 0.0
        is_dinner_window = 1.0 if 18 * 60 <= self.time_since_midnight < 20 * 60 else 0.0

        self.current_exercise_session = {
            "active": False,
            "remaining_steps": 0,
            "intensity": 0.0,
        }

        observed_state = np.array(
            [
                G0,
                self.previous_glucose_level,
                I_plasma0,
                time_since_last_meal,
                is_breakfast_window,
                is_lunch_window,
                is_dinner_window,
                time_since_last_exercise,
                self.current_exercise_session["intensity"],
                self.time_since_midnight,
            ],
            dtype=np.float32,
        )

        self.previous_observation = observed_state.copy()
        return observed_state, {}

    def build_observation(self):
        G0 = self.simulator_state[0]
        I_plasma0 = self.simulator_state[7]
        time_since_last_meal = self.simulator_state[5]
        time_since_last_exercise = self.simulator_state[6]

        self.time_since_midnight += self.dt

        is_breakfast_window = (
            1.0 if 6 * 60 <= self.time_since_midnight < 9 * 60 else 0.0
        )
        is_lunch_window = 1.0 if 12 * 60 <= self.time_since_midnight < 14 * 60 else 0.0
        is_dinner_window = 1.0 if 18 * 60 <= self.time_since_midnight < 20 * 60 else 0.0

        return np.array(
            [
                G0,
                self.previous_glucose_level,
                I_plasma0,
                time_since_last_meal,
                is_breakfast_window,
                is_lunch_window,
                is_dinner_window,
                time_since_last_exercise,
                self.simulator_state[4],
                self.time_since_midnight,
            ],
            dtype=np.float32,
        )

    def refresh_observation(self):
        G0 = self.simulator_state[0]
        I_plasma0 = self.simulator_state[7]
        time_since_last_meal = self.simulator_state[5]
        time_since_last_exercise = self.simulator_state[6]

        is_breakfast_window = (
            1.0 if 6 * 60 <= self.time_since_midnight < 9 * 60 else 0.0
        )
        is_lunch_window = 1.0 if 12 * 60 <= self.time_since_midnight < 14 * 60 else 0.0
        is_dinner_window = 1.0 if 18 * 60 <= self.time_since_midnight < 20 * 60 else 0.0

        obs = np.array(
            [
                G0,
                self.previous_glucose_level,
                I_plasma0,
                time_since_last_meal,
                is_breakfast_window,
                is_lunch_window,
                is_dinner_window,
                time_since_last_exercise,
                self.simulator_state[4],
                self.time_since_midnight,
            ],
            dtype=np.float32,
        )

        self.previous_observation = obs.copy()
        return obs

    def step(self, action):
        action = self.continuous_to_discrete(action)
        meal_category, exercise_mode, exercise_intensity, exercise_duration = action
        Rameal_current = self.map_meal_category_to_glucose_rate(meal_category)

        if self.random_events and np.random.rand() < 0.05:
            Rameal_current += np.random.uniform(5, 20) * 1000 / 60
            meal_event_occurred = True
        else:
            meal_event_occurred = meal_category > 0

        if exercise_mode == 1:
            if not self.current_exercise_session["active"]:
                self.current_exercise_session["active"] = True
                self.current_exercise_session["remaining_steps"] = exercise_duration
                self.current_exercise_session["intensity"] = (
                    self.map_exercise_level_to_intensity(exercise_intensity)
                )
        else:
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

        E_current = self.current_exercise_session["intensity"]

        if self.random_events and np.random.rand() < 0.03:
            E_current += np.random.uniform(0.2, 0.8)
            E_current = np.clip(E_current, 0.0, 1.0)
            exercise_event_occurred = True
        else:
            exercise_event_occurred = exercise_mode == 1

        ode_func = lambda t, y: self.simulator.odes(
            y, (Rameal_current, E_current, meal_event_occurred, exercise_event_occurred)
        )

        sol = solve_ivp(
            ode_func,
            t_span=(0, self.dt),
            y0=self.simulator_state,
            method="RK45",
            max_step=0.5,
        )

        self.simulator_state = sol.y[:, -1]
        obs = self.build_observation()
        obs[0] = self.add_observation_noise(obs[0])
        reward = self.compute_reward(action, obs)

        terminated = bool((obs[0] < 40) or (obs[0] > 450))
        truncated = obs[-1] >= 1440

        self.previous_glucose_level = obs[0]
        self.transition_log.append(
            {
                "state": self.previous_observation.copy(),
                "action": np.array(action),
                "reward": reward,
                "next_state": obs.copy(),
                "done": terminated or truncated,
            }
        )
        self.previous_observation = obs.copy()
        return obs, reward, terminated, truncated, {}

    def map_meal_category_to_glucose_rate(self, category):
        return [0.0, 0.45, 0.65, 0.8][category]

    def map_exercise_level_to_intensity(self, level):
        return [0.0, 0.3, 0.6, 0.9][level]

    def add_observation_noise(self, glucose_reading):
        return glucose_reading + np.random.normal(loc=0, scale=5.0)

    def compute_reward(self, action, obs):
        meal_category, exercise_mode = action[0], action[1]
        reward = 0.0
        time_since_last_meal = obs[3]
        windows = obs[4:7]
        if meal_category != 0 and not any(windows):
            reward += -50
        if meal_category != 0 and exercise_mode != 0:
            reward += -30
        glucose_level = obs[0]

        if time_since_last_meal > 480:
            reward += (
                100.0
                if 70 <= glucose_level <= 100
                else -30.0 if glucose_level <= 125 else -100.0
            )
        elif time_since_last_meal <= 120:
            reward += (
                80.0
                if glucose_level <= 140
                else -20.0 if glucose_level <= 199 else -100.0
            )
        else:
            reward += (
                50.0
                if 70 <= glucose_level <= 120
                else -15.0 if glucose_level <= 140 else -100.0
            )

        if meal_category == 0 and exercise_mode == 0 and 70 <= glucose_level <= 180:
            change = abs(glucose_level - self.previous_glucose_level)
            if change > 10:
                reward += -5 * (change - 10)

        if glucose_level < 40 or glucose_level > 450:
            reward -= 100

        return reward

    def toggle_random_events(self, enable):
        self.random_events = enable

    def clone_state(self):
        return {
            "simulator_state": self.simulator_state.copy(),
            "time_since_midnight": self.time_since_midnight,
            "previous_glucose_level": self.previous_glucose_level,
            "current_exercise_session": self.current_exercise_session.copy(),
            "previous_observation": self.previous_observation.copy(),
        }

    def restore_state(self, state_dict):
        self.simulator_state = state_dict["simulator_state"].copy()
        self.time_since_midnight = state_dict["time_since_midnight"]
        self.previous_glucose_level = state_dict["previous_glucose_level"]
        self.current_exercise_session = state_dict["current_exercise_session"].copy()
        self.previous_observation = state_dict["previous_observation"].copy()

    def force_action(
        self,
        meal_category=0,
        exercise_mode=0,
        exercise_intensity=0,
        exercise_duration=0,
    ):
        normalized_action = np.array(
            [
                meal_category / 3,
                exercise_mode,
                exercise_intensity / 3,
                exercise_duration / 12,
            ]
        )
        return self.step(normalized_action)
