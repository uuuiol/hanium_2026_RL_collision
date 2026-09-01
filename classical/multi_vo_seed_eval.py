"""
multi_vo_seed_eval.py

VO(Velocity Obstacle) 플래너를 MultiBoatEnv에서 평가합니다. multi_apf_seed_eval.py와
완전히 동일한 절차/포맷이고, 플래너만 다릅니다.

다른 배의 "속도"는 VO 계산에 필요한데 환경이 직접 주지 않으므로, 매 스텝
(이번 위치 - 직전 위치) / dt로 추정해서 씁니다 (직전 스텝에 그 배가 실제로 그렇게
움직였다는 사실 자체가 가장 정확한 속도 추정치이기 때문 - kinematic 모델이라 노이즈 없음).
"""
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common.multi_boat_env import MultiBoatEnv
from common.vo_agent import VOPlanner

SEEDS = [1, 2, 3]
N_AGENTS = 3
NUM_EPISODES_PER_SEED = 200
MAX_STEPS = 200
OUT_DIR = "results/multi_vo_seed_eval"
os.makedirs(OUT_DIR, exist_ok=True)

summary_rows = []

for seed in SEEDS:
    print(f"\n{'='*20} SEED {seed} {'='*20}", flush=True)
    env = MultiBoatEnv(n_agents=N_AGENTS, max_steps=MAX_STEPS, seed=seed)
    planner = VOPlanner(max_turn_rate=env.max_turn_rate, max_speed=env.max_speed)

    seed_dir = os.path.join(OUT_DIR, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    csv_path = os.path.join(seed_dir, f"vo_seed{seed}_results.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "avg_score", "goal_count", "collision_count",
                          "colreg_compliance_rate", "n_agents"])

    world_scores, world_goal_counts, world_collision_counts, world_colreg_rates = [], [], [], []

    for ep in range(1, NUM_EPISODES_PER_SEED + 1):
        states = env.reset()
        episode_rewards = [[] for _ in range(N_AGENTS)]
        prev_positions = [p.copy() for p in env.positions]
        # 배마다 "처음으로 끝난 시점"의 진짜 결과(goal/collision/timeout)를 저장.
        # env.step()은 이미 끝난 배에 대해 그 이후 매 스텝 "already_done"만 돌려주므로,
        # 루프가 끝난 뒤 마지막 스텝의 info만 보면 먼저 끝난 배들의 진짜 결과를 놓치게 됨.
        final_results = [None] * N_AGENTS

        while not env.all_done and env.step_count < MAX_STEPS:
            agent_velocities = [(env.positions[i] - prev_positions[i]) / env.dt for i in range(N_AGENTS)]

            actions = []
            for i in range(N_AGENTS):
                if env.done_flags[i]:
                    actions.append([0.0, 0.0])
                    continue
                obstacles = list(env.static_obstacles)
                obstacle_velocities = [np.zeros(2) for _ in env.static_obstacles]
                obstacle_radii = [env.obstacle_radius for _ in env.static_obstacles]
                for j in range(N_AGENTS):
                    if j == i or env.done_flags[j]:
                        continue
                    obstacles.append(env.positions[j])
                    obstacle_velocities.append(agent_velocities[j])
                    obstacle_radii.append(env.agent_radius)

                a = planner.get_action(env.positions[i], env.headings[i], env.goals[i],
                                        env.agent_radius, obstacles, obstacle_velocities, obstacle_radii)
                actions.append(a)

            prev_positions = [p.copy() for p in env.positions]
            next_states, rewards, terminateds, truncateds, infos = env.step(actions)
            for i in range(N_AGENTS):
                if infos[i]["result"] != "already_done":
                    episode_rewards[i].append(rewards[i])
                if final_results[i] is None and infos[i]["result"] not in ("running", "already_done"):
                    final_results[i] = infos[i]["result"]

        results = final_results
        n_goal = sum(1 for r in results if r == "goal")
        n_collision = sum(1 for r in results if "collision" in r)
        avg_score = np.mean([sum(episode_rewards[i]) for i in range(N_AGENTS)])
        colreg_rate = env.colreg_compliance_rate

        world_scores.append(avg_score)
        world_goal_counts.append(n_goal)
        world_collision_counts.append(n_collision)
        world_colreg_rates.append(colreg_rate)

        csv_writer.writerow([ep, avg_score, n_goal, n_collision,
                              "" if colreg_rate is None else colreg_rate, N_AGENTS])

        if ep % 50 == 0:
            colreg_str = "N/A" if colreg_rate is None else f"{colreg_rate*100:.0f}%"
            print(f"  seed{seed} ep {ep:4d}/{NUM_EPISODES_PER_SEED} | score {avg_score:6.2f} | "
                  f"goal {n_goal}/{N_AGENTS} | coll {n_collision}/{N_AGENTS} | colreg {colreg_str}", flush=True)

    csv_file.close()

    goal_rate = [c / N_AGENTS * 100 for c in world_goal_counts]
    collision_rate = [c / N_AGENTS * 100 for c in world_collision_counts]
    colreg_rate_pct = [r * 100 for r in world_colreg_rates if r is not None]

    overall_goal = float(np.mean(goal_rate))
    overall_coll = float(np.mean(collision_rate))
    overall_colreg = float(np.mean(colreg_rate_pct)) if colreg_rate_pct else float("nan")
    print(f"[SEED {seed} DONE] overall_goal_rate={overall_goal:.1f}% overall_collision_rate={overall_coll:.1f}% "
          f"overall_colreg_rate={overall_colreg:.1f}%", flush=True)
    summary_rows.append([seed, overall_goal, overall_coll, overall_colreg])

summary_path = os.path.join(OUT_DIR, "seed_sweep_summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["seed", "overall_goal_rate_pct", "overall_collision_rate_pct", "overall_colreg_rate_pct"])
    for row in summary_rows:
        w.writerow(row)

print(f"\n{'='*20} ALL SEEDS DONE {'='*20}", flush=True)
print(f"요약 저장: {summary_path}", flush=True)
for row in summary_rows:
    print(f"  seed{row[0]}: overall_goal={row[1]:.1f}% overall_coll={row[2]:.1f}% overall_colreg={row[3]:.1f}%", flush=True)
