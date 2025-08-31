import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from GlucoseDynamicsSimulator import GlucoseDynamicsSimulator


class CustomGlucoseDynamicsEnv(gym.Env):
    def __init__(
        self,
        random_events=True,
        weight=70.0,  # Can be float (fixed) or tuple (range)
        Gb=90.0,  # mg/dL
        Ib=5.7,
    ):  # µU/mL

        self.random_events = random_events  # toggle for stochastic events

        """
            Observation vector:
            [plasma_glucose_concentration, previous_glucose_concentration, plasma_insulin_concentration, time_since last meal, is_breakfast_window, is_lunch_window, is_dinner_window, time_since_last_exercise, current_activity_intensity, time_since_midnight]
        """
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array(
                [700.0, 700.0, 200.0, 1440.0, 1.0, 1.0, 1.0, 1440.0, 1.0, 1440.0]
            ),
            dtype=np.float32,
        )

        """
            self.action_space = spaces.Dict({
                "meal_category": spaces.Discrete(4),   # 0=no_meal, 1=light, 2=medium, 3=heavy
                "exercise_mode": spaces.Discrete(2),  # 0 = inactive, 1 = start/continue exercise
                "exercise_intensity": spaces.Discrete(4),  # 0 = rest, 1 = light, 2 = moderate, 3 = intense
                "exercise_duration": spaces.Discrete(13), # 0–60 min in 5-min steps
            })
        """
        self.action_space = spaces.MultiDiscrete([4, 2, 4, 13])  # flattened version

        """
            weight and glucose range from https://www.moh.gov.my/moh/resources/Penerbitan/CPG/Endocrine/3b.pdf
            insulin range from http://myjurnal.mohe.gov.my/public/article-view.php?id=3564#:~:text=The%20insulin%20sensitivity%20(HOMA%25S,%2C%20BMI%20and%20waist%20circumference).
        """
        self.weight = weight
        self.Gb = Gb
        self.Ib = Ib

        self.dt = 5  # 5 minutes

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

        temp_state = np.array(
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

        return temp_state

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

        return obs

    def step(self, action):
        meal_category = action[0]
        exercise_mode = action[1]
        exercise_intensity = action[2]
        exercise_duration = action[3]
        Rameal_current = self.map_meal_category_to_glucose_rate(meal_category)

        # random meal (increase environment instability)
        meal_event_occurred = meal_category > 0
        if self.random_events:
            if np.random.rand() < 0.05:
                random_meal_g = np.random.uniform(5, 20)
                random_meal_rate = random_meal_g * 1000 / 60
                Rameal_current += random_meal_rate
                meal_event_occurred = True

        if exercise_mode == 1:
            if not self.current_exercise_session["active"]:
                self.current_exercise_session["active"] = True
                self.current_exercise_session["remaining_steps"] = exercise_duration
                self.current_exercise_session["intensity"] = (
                    self.map_exercise_level_to_intensity(exercise_intensity)
                )
        else:
            self.current_exercise_session["active"] = False
            self.current_exercise_session["remaining_steps"] = 0
            self.current_exercise_session["intensity"] = 0.0

        if self.current_exercise_session["active"]:
            self.current_exercise_session["remaining_steps"] -= 1
            if self.current_exercise_session["remaining_steps"] <= 0:
                self.current_exercise_session["active"] = False
                self.current_exercise_session["intensity"] = 0.0

        E_current = self.current_exercise_session["intensity"]

        # random exercise (increase environment instability)
        exercise_event_occurred = exercise_mode == 1
        if self.random_events:
            if np.random.rand() < 0.03:
                random_exercise_intensity = np.random.uniform(0.2, 0.8)
                E_current += random_exercise_intensity
                E_current = float(np.clip(E_current, 0.0, 1.0))
                exercise_event_occurred = True

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

        # log transition
        self.transition_log.append(
            {
                "state": self.previous_observation.copy(),
                "action": (
                    action.copy() if isinstance(action, dict) else np.array(action)
                ),
                "reward": reward,
                "next_state": obs.copy(),
                "done": terminated or truncated,
            }
        )
        self.previous_observation = obs.copy()

        return obs, reward, terminated, truncated, {}

    def map_meal_category_to_glucose_rate(self, category):
        """
        References:
            American Diabetes Association, 2022 Standards of Care
            Diabetes Care. 2019;42(4):731–739. “Postprandial glucose: physiology and clinical significance.”
            OGTT kinetics: WHO/IDF guidelines on carbohydrate absorption
        """
        rates = [
            0.0,  # no meal
            0.45,  # light meal (30g carbs over 90 min)
            0.65,  # medium meal (60g carbs over 120 min)
            0.8,  # heavy meal (90g carbs over 150 min)
        ]
        return rates[int(category)]

    def map_exercise_level_to_intensity(self, level):
        """
        References:
            Ainsworth BE et al., 2011. Compendium of Physical Activities
            J Appl Physiol. 2017;122(4):889–901. “Glucose metabolism and exercise.”
            ADA 2022: exercise recommendations
        """
        intensities = [
            0.0,  # at rest
            0.3,  # light activity (2–3 METS)
            0.6,  # moderate activity (4–6 METS)
            0.9,  # intense (7–10 METS)
        ]
        return intensities[int(level)]

    def add_observation_noise(self, glucose_reading):
        """
        add noises into the glucose measurement to simulate the real life situation
        """
        noise = np.random.normal(loc=0, scale=5.0)
        return glucose_reading + noise

    def compute_reward(self, action, obs):
        meal_category = action[0]
        exercise_mode = action[1]

        # can be also calculated by scaling the penalty using the distance between glucose level and target glucose
        reward = 0.0

        # penalty for taking meal outside the meal windows
        time_since_last_meal = obs[3]  # From the state vector
        is_breakfast_window = obs[4]
        is_lunch_window = obs[5]
        is_dinner_window = obs[6]

        if meal_category != 0 and not (
            is_breakfast_window or is_lunch_window or is_dinner_window
        ):
            reward -= 50

        # action penalty
        if meal_category != 0 and exercise_mode != 0:
            reward += -30

        # state-dependent glucose reward or panalty
        glucose_level = obs[0]  # plasma glucose concentration

        if time_since_last_meal > 8 * 60:
            if 70 <= glucose_level <= 100:
                reward += 100.0
            elif 101 <= glucose_level <= 125:
                reward += -30.0
            elif glucose_level < 70:
                reward += -100.0
            elif glucose_level > 125:
                reward += -100.0

        elif time_since_last_meal <= 2 * 60:
            if glucose_level <= 140:
                reward += 80.0
            elif 141 <= glucose_level <= 199:
                reward += -20.0
            elif glucose_level < 70:
                reward += -100.0
            elif glucose_level > 200:
                reward += -100.0

        else:
            if 70 <= glucose_level <= 120:
                reward += 50.0
            elif 121 <= glucose_level <= 140:
                reward += -15.0
            elif glucose_level < 70:
                reward += -100.0
            elif glucose_level > 140:
                reward += -100.0

        # penalty for high variability when agent choose no meals and no exercise
        agent_inactive = meal_category == 0 and exercise_mode == 0

        if agent_inactive and (70 <= glucose_level <= 180):

            glucose_change = abs(glucose_level - self.previous_glucose_level)

            variability_threshold = 10.0
            variability_penalty_per_unit_over_threshold = -5

            if glucose_change > variability_threshold:
                reward += variability_penalty_per_unit_over_threshold * (
                    glucose_change - variability_threshold
                )

        # critical state penalty
        if glucose_level < 40 or glucose_level > 450:
            reward -= 100

        return reward

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

    def toggle_random_events(self, enable):
        self.random_events = enable

    def force_action(
        self,
        meal_category=0,
        exercise_mode=0,
        exercise_intensity=0,
        exercise_duration=0,
    ):
        """
        Force the environment to execute a specific action.

        :param meal_category: 0=no meal, 1=light, 2=medium, 3=heavy
        :param exercise_mode: 0=inactive, 1=exercise
        :param exercise_intensity: 0=rest, 1=light, 2=moderate, 3=intense
        :param exercise_duration: duration in discrete steps (0–12 corresponding to 0–60 minutes)
        """
        action = np.array(
            [meal_category, exercise_mode, exercise_intensity, exercise_duration],
            dtype=int,
        )
        return self.step(action)
