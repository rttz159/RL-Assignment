import streamlit as st
import numpy as np
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import time
from CustomGlucoseDynamicsEnv import CustomGlucoseDynamicsEnv
from DigitalTwinModel import DigitalTwinModel
from HybridLookaheadPolicyWrapper import HybridLookaheadPolicyWrapper
from stable_baselines3 import PPO

st.markdown(
    """
    <style>
    .stSidebar.st-emotion-cache-1lqf7hx.e1v5e29v0[data-testid="stSidebar"] {
        min-width: 400px !important;
        width: 400px !important;
    }

    .stMainBlockContainer.block-container.st-emotion-cache-1w723zb.e4man114[data-testid="stMainBlockContainer"] {
        max-width: 90% !important;   
        width: 90% !important;      
    }

    #interactive-glucose-simulation-real-time {
        text-align: center !important;
    }

    .stDataFrame[data-testid="stDataFrame"] thead tr th {
        position: sticky;
        top: 0;
        z-index: 2;
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def action_to_text(action, twin, step_minutes=5):
    meal_cat, mode, intensity, duration = twin._continuous_to_discrete(action)

    meal_texts = [
        "No meal (just water, tea, or fasting)",
        "Light snack – about 30g carbs (e.g., 1–2 slices of bread, a piece of fruit, or yogurt)",
        "Regular meal – about 60g carbs (e.g., rice or pasta plate, sandwich with sides, balanced lunch/dinner)",
        "Heavy meal – about 90g+ carbs (e.g., large pasta dish, fast food combo, festive or multi-course meal)",
    ]

    exercise_texts = [
        "Resting / relaxing",
        "Light activity (walking, light chores)",
        "Moderate activity (brisk walk, cycling)",
        "Intense activity (running, sports)",
    ]

    parts = []
    if meal_cat > 0:
        parts.append(meal_texts[meal_cat])
    if mode == 1:
        parts.append(f"{exercise_texts[intensity]} for {duration*step_minutes} min")
    if not parts:
        parts.append("Maintain routine")

    return ", ".join(parts)


st.title("Interactive Glucose Simulation")
st.caption("Select simulation parameters in the sidebar and click **Run Simulation**.")

st.sidebar.header("Simulation Parameters")

with st.sidebar.expander("User Physical Parameters", expanded=True):
    weight = st.slider("Weight (kg)", min_value=30, max_value=200, value=70, step=1)
    Gb = st.slider(
        "Baseline Glucose (mg/dL)", min_value=60, max_value=200, value=90, step=1
    )
    Ib = st.slider(
        "Insulin Basal Rate", min_value=0.0, max_value=20.0, value=5.7, step=0.1
    )

with st.sidebar.expander("Meal Schedule", expanded=True):
    st.write("Set meal times (24-hour format)")
    col1, col2, col3 = st.columns(3)
    with col1:
        breakfast = st.number_input("Breakfast", min_value=0, max_value=23, value=8)
    with col2:
        lunch = st.number_input("Lunch", min_value=0, max_value=23, value=13)
    with col3:
        dinner = st.number_input("Dinner", min_value=0, max_value=23, value=19)

with st.sidebar.expander("Extreme Glucose Events", expanded=False):
    st.write("Inject hypoglycemia or hyperglycemia at a given step")
    col1, col2 = st.columns(2)
    with col1:
        hypo_step = st.number_input(
            "Hypoglycemia Step", min_value=0, max_value=288, value=50
        )
    with col2:
        hyper_step = st.number_input(
            "Hyperglycemia Step", min_value=0, max_value=288, value=150
        )

with st.sidebar.expander("Simulation Speed", expanded=False):
    sim_speed = st.slider(
        "Step duration (seconds per simulation step)",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
    )

simulate_button = st.sidebar.button("Run Simulation", type="primary")

STEP_MINUTES = 5
SIMULATION_MINUTES = 1440
TOTAL_STEPS = SIMULATION_MINUTES // STEP_MINUTES
meal_schedule = {
    "breakfast": breakfast * 60,
    "lunch": lunch * 60,
    "dinner": dinner * 60,
}

if simulate_button:
    st.info("Running real-time simulation...")

    env = CustomGlucoseDynamicsEnv(random_events=False, weight=weight, Gb=Gb, Ib=Ib)
    obs, _ = env.reset()

    model = PPO.load("ppo_best_glucose_model.zip")
    twin = DigitalTwinModel(weight=weight, Gb=Gb, Ib=Ib, dt=STEP_MINUTES)
    planner = HybridLookaheadPolicyWrapper(
        sac_model=model,
        digital_twin=twin,
        horizon=6,
        candidates=5,
        target_glucose=100.0,
    )

    glucose_history = []
    time_history = []
    event_markers = []
    planner_predictions = []
    freeze_counter = 0
    injected_glucose = None
    freeze_steps = 3
    neutral_action = np.array([0, 0, 0, 0], dtype=np.float32)

    plot_container = st.empty()
    log_container = st.empty()
    log_data = []

    sns.set_style("whitegrid")
    plt.rcParams.update({"font.size": 12, "figure.dpi": 120})

    for step in range(TOTAL_STEPS):
        current_time = step * STEP_MINUTES

        if step == hypo_step:
            injected_glucose = 30.0
            env.simulator_state[0] = injected_glucose
            env.simulator_state[7] *= 2
            freeze_counter = freeze_steps
            event_markers.append((current_time, "hypoglycemia"))
            obs = env.refresh_observation()
        if step == hyper_step:
            injected_glucose = 300.0
            env.simulator_state[0] = injected_glucose
            env.simulator_state[5] = 0.0
            freeze_counter = freeze_steps
            event_markers.append((current_time, "hyperglycemia"))
            obs = env.refresh_observation()

        action, mode = planner.predict(
            obs, env_state=env.clone_state(), deterministic=True
        )
        suggested_text = action_to_text(action, twin)

        traj_obs, _ = twin.rollout_from_env_state(
            env.clone_state(), action, policy=model, horizon=6
        )
        planner_predictions.append((current_time, traj_obs[:, 0]))

        if freeze_counter > 0:
            obs[0] = injected_glucose
            freeze_counter -= 1
        else:
            for meal_name, meal_time in meal_schedule.items():
                if current_time == meal_time:
                    obs, _, _, _, _ = env.force_action(meal_category=3)
                    twin.sync_from_env_state(env.clone_state())
                    event_markers.append((current_time, meal_name))
                    break
            else:
                obs, _, _, _, _ = env.step(action)
                twin.sync_from_env_state(env.clone_state())

        glucose_history.append(obs[0])
        time_history.append(current_time)

        plt.figure(figsize=(16, 6))
        plt.plot(
            np.array(time_history) / 60,
            glucose_history,
            label="Actual Glucose",
            color="#1f77b4",
            linewidth=2,
        )
        for start_time, pred in planner_predictions:
            hours = (np.arange(len(pred)) * STEP_MINUTES + start_time) / 60
            plt.plot(
                hours,
                pred,
                color="#9467bd",
                linestyle="--",
                alpha=0.7,
                linewidth=1.5,
                label="_nolegend_",
            )

        pred_line = mlines.Line2D(
            [],
            [],
            color="#9467bd",
            linestyle="--",
            linewidth=1.5,
            label="Planner Predicted Trajectory",
        )

        plt.fill_between(
            np.array(time_history) / 60,
            planner.SAFE_LOW,
            planner.SAFE_HIGH,
            color="#d3f8d3",
            alpha=0.3,
            label="Safe Glucose Range",
        )

        plt.axhline(
            100, color="green", linestyle="--", linewidth=1.5, label="Target Glucose"
        )

        plt.axhline(
            planner.SAFE_LOW,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Hypoglycemia Threshold",
        )

        plt.axhline(
            planner.SAFE_HIGH,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            label="Hyperglycemia Threshold",
        )

        ymin, ymax = plt.ylim()
        label_height = ymax - 0.02 * (ymax - ymin)
        for t, label in event_markers:
            plt.axvline(t / 60, color="magenta", linestyle="--", alpha=0.6)
            plt.text(
                t / 60 - 0.2,
                label_height,
                label,
                rotation=90,
                color="magenta",
                fontsize=10,
                verticalalignment="top",
            )
        plt.xlabel("Time (24 hours)")
        plt.ylabel("Glucose (mg/dL)")
        plt.title("Real-Time Glucose Simulation", pad=36)

        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(pred_line)
        plt.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=len(handles),
            fontsize=10,
        )

        plt.grid(True)
        plt.tight_layout()
        plot_container.pyplot(plt)
        plt.close()

        event_label = ""

        if step == hypo_step:
            event_label = "Hypoglycemia injected"
        elif step == hyper_step:
            event_label = "Hyperglycemia injected"
        else:
            for meal_name, meal_time in meal_schedule.items():
                if current_time == meal_time:
                    event_label = meal_name
                    break

        log_data.append(
            {
                "Time": (
                    datetime(2000, 1, 1) + timedelta(minutes=current_time)
                ).strftime("%I:%M %p"),
                "Glucose (mg/dL)": "{:.2f}".format(round(obs[0], 1)),
                "Suggested Action": suggested_text,
                "Event": event_label,
            }
        )
        log_df = pd.DataFrame(log_data)
        log_container.dataframe(log_df[::-1], width="stretch", hide_index=True)

        time.sleep(sim_speed)

    st.success("Simulation complete!")
