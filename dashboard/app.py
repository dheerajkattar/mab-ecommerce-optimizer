"""Interactive Streamlit dashboard for live stats + simulator playground."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List

# Ensure project root is on path when run via streamlit run dashboard/app.py
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import plotly.graph_objects as go
import streamlit as st

from bandit_core.sim.synthetic import BernoulliBanditEnv
from bandit_core.state.memory import InMemoryStateStore
from bandit_core.strategies.base import BaseBanditStrategy
from bandit_core.strategies.epsilon_greedy import EpsilonGreedyStrategy
from bandit_core.strategies.thompson import ThompsonSamplingStrategy
from bandit_core.strategies.ucb1 import UCB1Strategy
from dashboard.redis_client import DashboardRedisClient

st.set_page_config(page_title="Bandit Dashboard", page_icon=":bar_chart:", layout="wide")

st.title("Multi-Armed Bandit Dashboard")
st.caption("Live experimentation telemetry + algorithm sandbox.")

redis_url = os.getenv("REDIS_URL", "redis://redis-db:6379/0")
default_refresh_seconds = float(os.getenv("LIVE_REFRESH_SECONDS", "3"))


def _build_strategy(name: str, seed: int) -> BaseBanditStrategy:
    store = InMemoryStateStore()
    normalized = name.strip().lower().replace("-", "_")
    if normalized == "thompson":
        return ThompsonSamplingStrategy(store=store, seed=seed)
    if normalized == "epsilon_greedy":
        return EpsilonGreedyStrategy(store=store, epsilon=0.1, seed=seed)
    if normalized == "ucb1":
        return UCB1Strategy(store=store, seed=seed)
    raise ValueError(f"Unsupported simulator strategy '{name}'")


def _extract_empirical_rate(state: Dict[str, float]) -> float:
    if "count" in state and state.get("count", 0.0) > 0:
        return state.get("value_sum", 0.0) / state["count"]
    if "alpha" in state and "beta" in state:
        alpha = state.get("alpha", 1.0)
        beta = state.get("beta", 1.0)
        denom = alpha + beta
        return alpha / denom if denom > 0 else 0.0
    return 0.0


live_tab, sim_tab = st.tabs(["Live Stats", "Simulator Playground"])

with live_tab:
    st.subheader("Live Stats")
    st.caption("Reads experiment/arm state directly from Redis.")

    try:
        redis_client = DashboardRedisClient(redis_url=redis_url)
        redis_client.ping()
    except Exception as exc:  # pragma: no cover - runtime environment dependent
        st.error(f"Unable to connect to Redis at `{redis_url}`: {exc}")
        st.stop()

    col1, col2 = st.columns([2, 1])
    with col1:
        experiment_ids = redis_client.list_experiment_ids()
        if not experiment_ids:
            st.info("No experiments found yet. Create one via the API first.")
        selected_experiment = st.selectbox(
            "Experiment",
            options=experiment_ids,
            index=0 if experiment_ids else None,
            placeholder="No experiments available",
        )
    with col2:
        auto_refresh = st.toggle("Auto refresh", value=False)
        refresh_seconds = st.number_input(
            "Refresh (seconds)",
            min_value=1.0,
            max_value=30.0,
            value=default_refresh_seconds,
            step=1.0,
        )

    if selected_experiment:
        experiment = redis_client.get_experiment(selected_experiment)
        if experiment is None:
            st.warning("Selected experiment was not found in Redis.")
        else:
            strategy_name = experiment.get("strategy") or "(default from API)"
            st.write(f"**Strategy:** `{strategy_name}`")
            st.write(f"**Arms:** `{', '.join(experiment['arm_ids'])}`")

            rows: List[Dict[str, float]] = []
            for arm_id in experiment["arm_ids"]:
                state = redis_client.get_arm_state(selected_experiment, arm_id)
                pulls = state.get("count", 0.0)
                successes = max(0.0, state.get("alpha", 1.0) - 1.0)
                failures = max(0.0, state.get("beta", 1.0) - 1.0)
                ts_pulls = successes + failures
                effective_pulls = pulls if pulls > 0 else ts_pulls
                rows.append(
                    {
                        "arm": arm_id,
                        "pulls": effective_pulls,
                        "empirical_rate": _extract_empirical_rate(state),
                    }
                )

            st.dataframe(rows, use_container_width=True, hide_index=True)

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[r["arm"] for r in rows],
                    y=[r["empirical_rate"] for r in rows],
                    text=[f"{r['empirical_rate']:.3f}" for r in rows],
                    textposition="auto",
                    name="Empirical rate",
                )
            )
            fig.update_layout(
                title="Arm empirical conversion rates",
                xaxis_title="Arm",
                yaxis_title="Conversion rate",
                yaxis=dict(range=[0, 1]),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

    if auto_refresh:
        time.sleep(refresh_seconds)
        st.rerun()

with sim_tab:
    st.subheader("Simulator Playground")
    st.caption("Adjust true conversion rates and watch cumulative regret evolve.")

    setup_col, control_col = st.columns([2, 1])
    with setup_col:
        arm_a_rate = st.slider("True conversion: Arm A", min_value=0.0, max_value=1.0, value=0.12)
        arm_b_rate = st.slider("True conversion: Arm B", min_value=0.0, max_value=1.0, value=0.18)
        arm_c_rate = st.slider("True conversion: Arm C", min_value=0.0, max_value=1.0, value=0.09)
    with control_col:
        strategy_label = st.selectbox(
            "Algorithm",
            options=["ucb1", "thompson", "epsilon_greedy"],
            index=0,
        )
        rounds = st.slider("Rounds", min_value=50, max_value=5000, value=1000, step=50)
        update_every = st.slider("Animation step", min_value=10, max_value=200, value=25, step=5)
        seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1)

    run_clicked = st.button("Run simulation")
    if run_clicked:
        arm_rates = {"A": arm_a_rate, "B": arm_b_rate, "C": arm_c_rate}
        best_rate = max(arm_rates.values())
        seed_int = int(seed)

        strategy = _build_strategy(strategy_label, seed=seed_int)
        env = BernoulliBanditEnv(arm_rates=arm_rates, seed=seed_int)
        experiment_id = "sim_playground"
        arm_ids = list(arm_rates.keys())
        strategy.initialize_experiment(experiment_id, arm_ids)

        cumulative_regret = 0.0
        cumulative_reward = 0.0
        regrets: List[float] = []
        rewards: List[float] = []
        chosen_counts = {arm: 0 for arm in arm_ids}

        line_placeholder = st.empty()
        bar_placeholder = st.empty()
        status_placeholder = st.empty()

        for t in range(1, rounds + 1):
            arm = strategy.select_arm(experiment_id, arm_ids)
            reward = env.pull(arm)
            strategy.update(experiment_id, arm, reward)
            chosen_counts[arm] += 1

            cumulative_regret += best_rate - arm_rates[arm]
            cumulative_reward += reward
            regrets.append(cumulative_regret)
            rewards.append(cumulative_reward)

            should_draw = (t % update_every == 0) or (t == rounds)
            if should_draw:
                rounds_axis = list(range(1, t + 1))

                line_fig = go.Figure()
                line_fig.add_trace(go.Scatter(x=rounds_axis, y=regrets, mode="lines", name="Cumulative regret"))
                line_fig.add_trace(go.Scatter(x=rounds_axis, y=rewards, mode="lines", name="Cumulative reward"))
                line_fig.update_layout(
                    title=f"Learning curves ({strategy_label})",
                    xaxis_title="Round",
                    yaxis_title="Value",
                    height=420,
                )
                line_placeholder.plotly_chart(line_fig, use_container_width=True)

                bar_fig = go.Figure()
                bar_fig.add_trace(
                    go.Bar(
                        x=list(chosen_counts.keys()),
                        y=list(chosen_counts.values()),
                        text=list(chosen_counts.values()),
                        textposition="auto",
                    )
                )
                bar_fig.update_layout(
                    title="Arm selection counts",
                    xaxis_title="Arm",
                    yaxis_title="Selections",
                    height=320,
                )
                bar_placeholder.plotly_chart(bar_fig, use_container_width=True)

                status_placeholder.write(
                    f"Round {t}/{rounds} | "
                    f"Cumulative regret: {cumulative_regret:.2f} | "
                    f"Cumulative reward: {cumulative_reward:.2f}"
                )
