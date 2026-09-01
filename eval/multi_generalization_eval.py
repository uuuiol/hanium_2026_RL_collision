"""
multi_generalization_eval.py

지금까지 학습/평가는 전부 "고정된 시작-목표 5쌍" 시나리오 하나에서만 이뤄졌습니다.
이 스크립트는 매 에피소드 월드 전체 범위에서 시작/목표를 무작위로 새로 뽑아서
(env.reset(randomize=True)), 학습 때 한 번도 못 본 배치에서도 정책이 잘 작동하는지
("일반화") 확인합니다.

- PPO/TD3/SAC: 시드스윕에서 학습한 best 체크포인트를 그리디(노이즈 없이)로 평가
- DDPG: 참고용으로 포함하되, 애초에 고정 시나리오에서도 불안정했던 알고리즘이라 기대치는 낮음
- APF/VO: 재학습 필요 없는 규칙기반 컨트롤러라 그대로 평가 (오히려 이쪽이 "설계상 원래
  일반화되는" 방법이라 비교 기준이 됨)
"""
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from common.multi_boat_env import MultiBoatEnv
from common.ddpg_model import Actor as DDPGActor
from sac.sac_model import GaussianActor
from ppo.ppo_model import ActorCritic
from common.apf_agent import APFPlanner
from common.vo_agent import VOPlanner

N_AGENTS = 3
MAX_STEPS = 200
NUM_EVAL_EPISODES = 100
SEEDS = [1, 2, 3]
OUT_DIR = "results/multi_generalization_eval"
os.makedirs(OUT_DIR, exist_ok=True)

RL_ALGOS = {
    "ddpg": "results/multi_ddpg_seed_sweep/seed{seed}/multi_ddpg_ep600_seed{seed}_actor_weights_best.weights.h5",
    "td3": "results/multi_td3_seed_sweep/seed{seed}/multi_td3_ep600_seed{seed}_actor_weights_best.weights.h5",
    "sac": "results/multi_sac_seed_sweep/seed{seed}/multi_sac_ep600_seed{seed}_actor_weights_best.weights.h5",
    "ppo": "results/multi_ppo_seed_sweep_128relu/seed{seed}/multi_ppo_ep600_seed{seed}_model_weights_best.weights.h5",
}


def build_rl_greedy_fn(algo, state_size, action_size, weights_path):
    dummy_s = np.zeros([1, state_size], dtype=np.float32)
    if algo in ("ddpg", "td3"):
        net = DDPGActor(state_size, action_size)
        net(dummy_s)
        net.load_weights(weights_path)
        return lambda state: np.clip(net(np.reshape(state, [1, -1]).astype(np.float32)).numpy()[0], -1.0, 1.0)

    if algo == "sac":
        net = GaussianActor(state_size, action_size)
        net(dummy_s)
        net.load_weights(weights_path)

        def greedy(state):
            _, _, det_action = net.sample(np.reshape(state, [1, -1]).astype(np.float32))
            return np.clip(det_action.numpy()[0], -1.0, 1.0)
        return greedy

    if algo == "ppo":
        net = ActorCritic(state_size, action_size)
        net(dummy_s)
        net.load_weights(weights_path)

        def greedy(state):
            mu, _ = net(np.reshape(state, [1, -1]).astype(np.float32))
            return np.clip(mu.numpy()[0], -1.0, 1.0)
        return greedy

    raise ValueError(algo)


def run_rl_eval(algo, seed, greedy_fn):
    env = MultiBoatEnv(n_agents=N_AGENTS, max_steps=MAX_STEPS, seed=seed)
    goal_counts, collision_counts, colreg_rates = [], [], []
    for ep in range(NUM_EVAL_EPISODES):
        states = env.reset(randomize=True)
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
    return goal_counts, collision_counts, colreg_rates


def run_planner_eval(algo, seed):
    env = MultiBoatEnv(n_agents=N_AGENTS, max_steps=MAX_STEPS, seed=seed)
    if algo == "apf":
        planner = APFPlanner(max_turn_rate=env.max_turn_rate, max_speed=env.max_speed)
    else:
        planner = VOPlanner(max_turn_rate=env.max_turn_rate, max_speed=env.max_speed)

    goal_counts, collision_counts, colreg_rates = [], [], []
    for ep in range(NUM_EVAL_EPISODES):
        env.reset(randomize=True)
        prev_positions = [p.copy() for p in env.positions]
        final_results = [None] * N_AGENTS

        while not env.all_done and env.step_count < MAX_STEPS:
            agent_velocities = [(env.positions[i] - prev_positions[i]) / env.dt for i in range(N_AGENTS)]
            actions = []
            for i in range(N_AGENTS):
                if env.done_flags[i]:
                    actions.append([0.0, 0.0])
                    continue
                if algo == "apf":
                    obstacles = [(o, env.obstacle_radius) for o in env.static_obstacles]
                    for j in range(N_AGENTS):
                        if j == i or env.done_flags[j]:
                            continue
                        obstacles.append((env.positions[j], env.agent_radius))
                    a = planner.get_action(env.positions[i], env.headings[i], env.goals[i], obstacles)
                else:
                    obstacles = list(env.static_obstacles)
                    obs_vels = [np.zeros(2) for _ in env.static_obstacles]
                    obs_radii = [env.obstacle_radius for _ in env.static_obstacles]
                    for j in range(N_AGENTS):
                        if j == i or env.done_flags[j]:
                            continue
                        obstacles.append(env.positions[j])
                        obs_vels.append(agent_velocities[j])
                        obs_radii.append(env.agent_radius)
                    a = planner.get_action(env.positions[i], env.headings[i], env.goals[i],
                                            env.agent_radius, obstacles, obs_vels, obs_radii)
                actions.append(a)

            prev_positions = [p.copy() for p in env.positions]
            _, rewards, terminateds, truncateds, infos = env.step(actions)
            for i in range(N_AGENTS):
                if final_results[i] is None and infos[i]["result"] not in ("running", "already_done"):
                    final_results[i] = infos[i]["result"]

        goal_counts.append(sum(1 for r in final_results if r == "goal"))
        collision_counts.append(sum(1 for r in final_results if r is not None and "collision" in r))
        colreg_rates.append(env.colreg_compliance_rate)
    return goal_counts, collision_counts, colreg_rates


summary_rows = []

for algo, pattern in RL_ALGOS.items():
    print(f"\n{'='*20} {algo.upper()} {'='*20}", flush=True)
    for seed in SEEDS:
        weights_path = pattern.format(seed=seed)
        if not os.path.exists(weights_path):
            print(f"  seed{seed}: 가중치 없음 - 건너뜀", flush=True)
            continue
        env_probe = MultiBoatEnv(n_agents=N_AGENTS, max_steps=MAX_STEPS, seed=seed)
        greedy_fn = build_rl_greedy_fn(algo, env_probe.state_size, env_probe.action_size, weights_path)
        goal_counts, collision_counts, colreg_rates = run_rl_eval(algo, seed, greedy_fn)

        goal_rate = np.mean([c / N_AGENTS * 100 for c in goal_counts])
        collision_rate = np.mean([c / N_AGENTS * 100 for c in collision_counts])
        colreg_pct = [r * 100 for r in colreg_rates if r is not None]
        colreg_rate = np.mean(colreg_pct) if colreg_pct else float("nan")
        print(f"  seed{seed}: goal_rate={goal_rate:.1f}% collision_rate={collision_rate:.1f}% "
              f"colreg_rate={colreg_rate:.1f}%", flush=True)
        summary_rows.append([algo, seed, goal_rate, collision_rate, colreg_rate])

for algo in ["apf", "vo"]:
    print(f"\n{'='*20} {algo.upper()} {'='*20}", flush=True)
    for seed in SEEDS:
        goal_counts, collision_counts, colreg_rates = run_planner_eval(algo, seed)
        goal_rate = np.mean([c / N_AGENTS * 100 for c in goal_counts])
        collision_rate = np.mean([c / N_AGENTS * 100 for c in collision_counts])
        colreg_pct = [r * 100 for r in colreg_rates if r is not None]
        colreg_rate = np.mean(colreg_pct) if colreg_pct else float("nan")
        print(f"  seed{seed}: goal_rate={goal_rate:.1f}% collision_rate={collision_rate:.1f}% "
              f"colreg_rate={colreg_rate:.1f}%", flush=True)
        summary_rows.append([algo, seed, goal_rate, collision_rate, colreg_rate])

summary_path = os.path.join(OUT_DIR, "generalization_summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["algo", "seed", "goal_rate_pct", "collision_rate_pct", "colreg_rate_pct"])
    for row in summary_rows:
        w.writerow(row)

print(f"\n{'='*20} ALL DONE {'='*20}", flush=True)
print(f"요약 저장: {summary_path}", flush=True)

by_algo = {}
for algo, seed, g, c, cr in summary_rows:
    by_algo.setdefault(algo, {"goal": [], "coll": [], "colreg": []})
    by_algo[algo]["goal"].append(g)
    by_algo[algo]["coll"].append(c)
    if not np.isnan(cr):
        by_algo[algo]["colreg"].append(cr)

for algo, d in by_algo.items():
    g = np.mean(d["goal"])
    c = np.mean(d["coll"])
    cr = np.mean(d["colreg"]) if d["colreg"] else float("nan")
    print(f"[{algo.upper()}] 평균: goal_rate={g:.1f}% collision_rate={c:.1f}% colreg_rate={cr:.1f}%", flush=True)
