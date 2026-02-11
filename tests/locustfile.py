# Locust load-testing file for the MAB e-commerce optimizer API.
#
# Web UI (default):  docker compose run --rm load-tester
#                    Then open http://localhost:8089
#
# Headless run (copy this single line; image entrypoint is already locust):
#   docker compose run --rm load-tester -f /mnt/locust/locustfile.py --host http://bandit-api:8000 --headless -u 100 -r 10 -t 60s --csv /mnt/locust/results

import random

from locust import HttpUser, between, task


class BanditUser(HttpUser):
    """Simulates a visitor that asks the bandit for a decision and
    occasionally sends back a reward signal."""

    wait_time = between(0.5, 2)

    EXPERIMENT_ID = "load_test"
    ARM_IDS = ["A", "B"]
    # Ground-truth reward probabilities per arm.
    REWARD_PROBS = {"A": 0.05, "B": 0.20}

    def on_start(self):
        """Seed the experiment once; ignore 409 if another user already created it."""
        self.last_arm_id = None
        with self.client.post(
            "/experiments",
            json={
                "experiment_id": self.EXPERIMENT_ID,
                "arm_ids": self.ARM_IDS,
                "strategy": "THOMPSON",
            },
            catch_response=True,
        ) as resp:
            if resp.status_code in (201, 409):
                resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(5)
    def get_decision(self):
        """Ask the bandit which arm to show (high frequency)."""
        with self.client.get(
            "/decision",
            params={"experiment_id": self.EXPERIMENT_ID},
            name="/decision",
            catch_response=True,
        ) as resp:
            if resp.ok:
                self.last_arm_id = resp.json()["arm_id"]

    @task(1)
    def send_reward(self):
        """Report a conversion signal back to the bandit (lower frequency)."""
        if self.last_arm_id is None:
            return

        arm = self.last_arm_id
        reward = 1.0 if random.random() < self.REWARD_PROBS.get(arm, 0.0) else 0.0

        self.client.post(
            "/reward",
            json={
                "experiment_id": self.EXPERIMENT_ID,
                "arm_id": arm,
                "reward": reward,
            },
            name="/reward",
        )
        self.last_arm_id = None
