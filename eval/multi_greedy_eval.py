"""
multi_greedy_eval.py

지금까지 시드스윕(run_multi_*_seeds.py)에서 학습한 각 알고리즘의 "최고성능 시점(best)"
가중치를 불러와서, 탐험 노이즈/엔트로피 없이 "그리디(결정론적) 행동"으로만 평가합니다.

왜 필요한가:
- 학습 중 CSV에 기록된 목표도달률/충돌률은 DDPG/TD3의 감쇠 노이즈, SAC의 엔트로피 샘플링,
  PPO의 확률적 정책이 전부 낀 채로 잰 값입니다 - "학습이 얼마나 잘 진행됐는지"는 보여주지만
  "최종 정책이 실제로 얼마나 잘하는지"는 아닙니다.
- 여기서는 각 알고리즘의 결정론적 행동(DDPG/TD3는 actor(s) 그대로, SAC는 tanh(mu), PPO는
  action mean)만 써서, 알고리즘 간 최종 성능을 노이즈 없이 공정하게 비교합니다.
- APF/VO는 원래부터 확률성이 없는 결정론적 컨트롤러라 재평가할 필요 없음
  (multi_apf_seed_eval.py / multi_vo_seed_eval.py 결과가 이미 그리디 평가입니다).

버그 수정: multi_boat_env.py의 step()은 이미 끝난 배에 대해 그 이후 매 스텝 "already_done"만
돌려주므로, 에피소드 끝난 뒤 "마지막 스텝 info"만 보면 먼저 끝난 배의 결과를 놓칩니다.
여기서도 매 스텝 "처음 끝난 시점의 진짜 결과"를 따로 저장합니다.
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

N_AGENTS = 3
MAX_STEPS = 200
NUM_EVAL_EPISODES = 100
SEEDS = [1, 2, 3]
OUT_DIR = "results/multi_greedy_eval"
os.makedirs(OUT_DIR, exist_ok=True)

# (알고리즘 이름, 가중치 경로 패턴, 로더 함수) 등록
ALGOS = {
    "ddpg": {
        "weights_pattern": "results/multi_ddpg_seed_sweep/seed{seed}/multi_ddpg_ep600_seed{seed}_actor_weights_best.weights.h5",
    },
    "td3": {
        "weights_pattern": "results/multi_td3_seed_sweep/seed{seed}/multi_td3_ep600_seed{seed}_actor_weights_best.weights.h5",
    },
    "sac": {
        "weights_pattern": "results/multi_sac_seed_sweep/seed{seed}/multi_sac_ep600_seed{seed}_actor_weights_best.weights.h5",
    },
    "ppo": {
        "weights_pattern": "results/multi_ppo_seed_sweep_128relu/seed{seed}/multi_ppo_ep600_seed{seed}_model_weights_best.weights.h5",
    },
}


def build_greedy_fn(algo, state_size, action_size, weights_path):
    dummy_s = np.zeros([1, state_size], dtype=np.float32)
    if algo in ("ddpg", "td3"):
        net = DDPGActor(state_size, action_size)
        net(dummy_s)
        net.load_weights(weights_path)

        def greedy(state):
            state_in = np.reshape(state, [1, -1]).astype(np.float32)
            return np.clip(net(state_in).numpy()[0], -1.0, 1.0)
        return greedy

    if algo == "sac":
        net = GaussianActor(state_size, action_size)
        net(dummy_s)
        net.load_weights(weights_path)

        def greedy(state):
            state_in = np.reshape(state, [1, -1]).astype(np.float32)
            _, _, det_action = net.sample(state_in)
            return np.clip(det_action.numpy()[0], -1.0, 1.0)
        return greedy

    if algo == "ppo":
        net = ActorCritic(state_size, action_size)
        net(dummy_s)
        net.load_weights(weights_path)

        def greedy(state):
            state_in = np.reshape(state, [1, -1]).astype(np.float32)
            mu, _ = net(state_in)
            return np.clip(mu.numpy()[0], -1.0, 1.0)
        return greedy

    raise ValueError(algo)


def run_eval(algo, seed, greedy_fn):
    env = MultiBoatEnv(n_agents=N_AGENTS, max_steps=MAX_STEPS, seed=seed)

    goal_counts, collision_counts, colreg_rates = [], [], []
    for ep in range(NUM_EVAL_EPISODES):
        states = env.reset()
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

        n_goal = sum(1 for r in final_results if r == "goal")
        n_collision = sum(1 for r in final_results if r is not None and "collision" in r)
        goal_counts.append(n_goal)
        collision_counts.append(n_collision)
        colreg_rates.append(env.colreg_compliance_rate)

    goal_rate = np.mean([c / N_AGENTS * 100 for c in goal_counts])
    collision_rate = np.mean([c / N_AGENTS * 100 for c in collision_counts])
    colreg_pct = [r * 100 for r in colreg_rates if r is not None]
    colreg_rate = np.mean(colreg_pct) if colreg_pct else float("nan")
    return goal_rate, collision_rate, colreg_rate


summary_rows = []
for algo, cfg in ALGOS.items():
    print(f"\n{'='*20} {algo.upper()} {'='*20}", flush=True)
    for seed in SEEDS:
        weights_path = cfg["weights_pattern"].format(seed=seed)
        if not os.path.exists(weights_path):
            print(f"  seed{seed}: 가중치 없음({weights_path}) - 건너뜀", flush=True)
            continue

        env_probe = MultiBoatEnv(n_agents=N_AGENTS, max_steps=MAX_STEPS, seed=seed)
        greedy_fn = build_greedy_fn(algo, env_probe.state_size, env_probe.action_size, weights_path)

        goal_rate, collision_rate, colreg_rate = run_eval(algo, seed, greedy_fn)
        print(f"  seed{seed}: goal_rate={goal_rate:.1f}% collision_rate={collision_rate:.1f}% "
              f"colreg_rate={colreg_rate:.1f}%", flush=True)
        summary_rows.append([algo, seed, goal_rate, collision_rate, colreg_rate])

summary_path = os.path.join(OUT_DIR, "greedy_eval_summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["algo", "seed", "goal_rate_pct", "collision_rate_pct", "colreg_rate_pct"])
    for row in summary_rows:
        w.writerow(row)

print(f"\n{'='*20} ALL DONE {'='*20}", flush=True)
print(f"요약 저장: {summary_path}", flush=True)

# 알고리즘별 평균
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
