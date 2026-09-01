"""
multi_generalization_eval_v2.py

도메인 랜덤화로 재학습한 TD3/SAC/PPO 가중치를, 기존(고정 시나리오로 학습한) 가중치와
같은 기준으로 비교합니다. 각 모델을 "고정 시나리오"와 "랜덤 시나리오" 둘 다에서
그리디로 평가해서, 도메인 랜덤화가 (1) 일반화를 실제로 개선했는지 (2) 원래 잘하던
고정 시나리오 성능을 얼마나 희생했는지 확인합니다.
"""
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.multi_boat_env import MultiBoatEnv
from common.ddpg_model import Actor as DDPGActor
from sac.sac_model import GaussianActor
from ppo.ppo_model import ActorCritic

N_AGENTS = 3
MAX_STEPS = 200
NUM_EVAL_EPISODES = 100
SEEDS = [1, 2, 3]
OUT_DIR = "results/multi_generalization_eval_v2"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_SETS = {
    "sac_fixed":  ("sac", "results/multi_sac_seed_sweep/seed{seed}/multi_sac_ep600_seed{seed}_actor_weights_best.weights.h5"),
    "sac_rand":   ("sac", "results/multi_sac_seed_sweep_randomized/seed{seed}/multi_sac_ep600_rand_seed{seed}_actor_weights_best.weights.h5"),
    "ppo_fixed":  ("ppo", "results/multi_ppo_seed_sweep_128relu/seed{seed}/multi_ppo_ep600_seed{seed}_model_weights_best.weights.h5"),
    "ppo_rand":   ("ppo", "results/multi_ppo_seed_sweep_randomized/seed{seed}/multi_ppo_ep600_rand_seed{seed}_model_weights_best.weights.h5"),
}


def build_greedy_fn(algo, state_size, action_size, weights_path):
    dummy_s = np.zeros([1, state_size], dtype=np.float32)
    if algo == "td3":
        net = DDPGActor(state_size, action_size)
        net(dummy_s)
        net.load_weights(weights_path)
        return lambda s: np.clip(net(np.reshape(s, [1, -1]).astype(np.float32)).numpy()[0], -1.0, 1.0)
    if algo == "sac":
        net = GaussianActor(state_size, action_size)
        net(dummy_s)
        net.load_weights(weights_path)

        def greedy(s):
            _, _, det = net.sample(np.reshape(s, [1, -1]).astype(np.float32))
            return np.clip(det.numpy()[0], -1.0, 1.0)
        return greedy
    if algo == "ppo":
        net = ActorCritic(state_size, action_size)
        net(dummy_s)
        net.load_weights(weights_path)

        def greedy(s):
            mu, _ = net(np.reshape(s, [1, -1]).astype(np.float32))
            return np.clip(mu.numpy()[0], -1.0, 1.0)
        return greedy
    raise ValueError(algo)


def run_eval(seed, greedy_fn, randomize):
    env = MultiBoatEnv(n_agents=N_AGENTS, max_steps=MAX_STEPS, seed=seed)
    goal_counts, collision_counts, colreg_rates = [], [], []
    for ep in range(NUM_EVAL_EPISODES):
        states = env.reset(randomize=randomize)
        final_results = [None] * N_AGENTS
        while not env.all_done and env.step_count < MAX_STEPS:
            actions = []
            for i in range(N_AGENTS):
                if env.done_flags[i]:
                    actions.append([0.0, 0.0])
                    continue
                actions.append(greedy_fn(states[i]))
            states, rewards, terminateds, truncateds, infos = env.step(actions)
            for i in range(N_AGENTS):
                if final_results[i] is None and infos[i]["result"] not in ("running", "already_done"):
                    final_results[i] = infos[i]["result"]
        goal_counts.append(sum(1 for r in final_results if r == "goal"))
        collision_counts.append(sum(1 for r in final_results if r is not None and "collision" in r))
        colreg_rates.append(env.colreg_compliance_rate)
    goal_rate = np.mean([c / N_AGENTS * 100 for c in goal_counts])
    collision_rate = np.mean([c / N_AGENTS * 100 for c in collision_counts])
    colreg_pct = [r * 100 for r in colreg_rates if r is not None]
    colreg_rate = np.mean(colreg_pct) if colreg_pct else float("nan")
    return goal_rate, collision_rate, colreg_rate


summary_rows = []
for model_name, (algo, pattern) in MODEL_SETS.items():
    print(f"\n{'='*20} {model_name} {'='*20}", flush=True)
    for scenario, randomize in [("fixed", False), ("random", True)]:
        goals, colls, colregs = [], [], []
        for seed in SEEDS:
            weights_path = pattern.format(seed=seed)
            if not os.path.exists(weights_path):
                print(f"  [{scenario}] seed{seed}: 가중치 없음({weights_path}) - 건너뜀", flush=True)
                continue
            env_probe = MultiBoatEnv(n_agents=N_AGENTS, max_steps=MAX_STEPS, seed=seed)
            greedy_fn = build_greedy_fn(algo, env_probe.state_size, env_probe.action_size, weights_path)
            g, c, cr = run_eval(seed, greedy_fn, randomize)
            print(f"  [{scenario}] seed{seed}: goal={g:.1f}% coll={c:.1f}% colreg={cr:.1f}%", flush=True)
            goals.append(g)
            colls.append(c)
            if not np.isnan(cr):
                colregs.append(cr)
            summary_rows.append([model_name, scenario, seed, g, c, cr])
        if goals:
            print(f"  [{model_name}/{scenario}] 평균: goal={np.mean(goals):.1f}% coll={np.mean(colls):.1f}% "
                  f"colreg={np.mean(colregs) if colregs else float('nan'):.1f}%", flush=True)

summary_path = os.path.join(OUT_DIR, "v2_summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "scenario", "seed", "goal_rate_pct", "collision_rate_pct", "colreg_rate_pct"])
    for row in summary_rows:
        w.writerow(row)
print(f"\n요약 저장: {summary_path}", flush=True)
