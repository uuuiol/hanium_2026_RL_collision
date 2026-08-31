"""
watch_multi_ddpg_trained_agents.py

multi_ddpg_train_logging.py로 학습한 DDPG 정책을 불러와서 화면에 실시간으로 보여줍니다.
watch_multi_trained_agents.py(PPO용)와 완전히 동일한 절차이고, 신경망만 Actor로 바뀝니다.
"""
import time
import numpy as np
from collections import Counter

from multi_boat_env import MultiBoatEnv
from ddpg_model import Actor

# ===== 설정 =====
WEIGHTS_PATH = "multi_ddpg_actor_weights_best.weights.h5"
N_AGENTS = 3
MAX_STEPS = 200

NUM_SCAN_EPISODES = 30
TOP_N = 5
RENDER_PAUSE = 0.03
# ================

AGENT_COLOR_NAMES = ["빨강", "보라", "주황", "갈색", "검정"]

env = MultiBoatEnv(n_agents=N_AGENTS, max_steps=MAX_STEPS)

actor = Actor(env.state_size, env.action_size)
actor(np.zeros([1, env.state_size], dtype=np.float32))
actor.load_weights(WEIGHTS_PATH)
print(f"모델 가중치 불러오기 완료: {WEIGHTS_PATH}")


def get_greedy_action(state):
    state_in = np.reshape(state, [1, -1]).astype(np.float32)
    mu = actor(state_in).numpy()[0]
    return np.clip(mu, -1.0, 1.0)


def snapshot(step_count):
    return {
        "positions": [p.copy() for p in env.positions],
        "headings": list(env.headings),
        "done_flags": list(env.done_flags),
        "step_count": step_count,
    }


# ===== 1단계: 화면 없이 빠르게 여러 판 스캔 =====
print(f"\n{NUM_SCAN_EPISODES}판을 미리 빠르게 돌려보는 중...")
scanned_episodes = []

for ep in range(NUM_SCAN_EPISODES):
    states = env.reset()
    frames = [snapshot(0)]
    total_scores = [0.0] * N_AGENTS
    final_agent_results = [None] * N_AGENTS

    while not env.all_done and env.step_count < MAX_STEPS:
        actions = []
        for i in range(N_AGENTS):
            if env.done_flags[i]:
                actions.append([0.0, 0.0])
                continue
            actions.append(get_greedy_action(states[i]))

        states, rewards, terminateds, truncateds, infos = env.step(actions)
        for i in range(N_AGENTS):
            total_scores[i] += rewards[i]
            if final_agent_results[i] is None and infos[i]["result"] not in ("running", "already_done"):
                final_agent_results[i] = infos[i]["result"]
        frames.append(snapshot(env.step_count))

    for i in range(N_AGENTS):
        if final_agent_results[i] is None:
            final_agent_results[i] = "timeout"

    agent_results = final_agent_results
    n_goal = sum(1 for r in agent_results if r == "goal")
    avg_score = float(np.mean(total_scores))

    scanned_episodes.append({
        "scan_index": ep + 1,
        "score": avg_score,
        "n_goal": n_goal,
        "agent_results": agent_results,
        "frames": frames,
    })
    print(f"  [{ep + 1:3d}/{NUM_SCAN_EPISODES}] 목표도달 {n_goal}/{N_AGENTS} | score: {avg_score:7.2f} "
         f"| 배별 결과: {agent_results}")

# ===== 스캔 결과 요약 =====
goal_dist = Counter(e["n_goal"] for e in scanned_episodes)
print(f"\n이번 스캔 {NUM_SCAN_EPISODES}판의 목표도달 분포:")
for k in range(N_AGENTS + 1):
    cnt = goal_dist.get(k, 0)
    print(f"  {k}척 성공: {cnt}판 ({cnt / NUM_SCAN_EPISODES * 100:.0f}%)")

# ===== 2단계: 정렬 =====
scanned_episodes.sort(key=lambda e: (e["n_goal"], e["score"]), reverse=True)
top_episodes = scanned_episodes[:TOP_N]

best_n_goal = top_episodes[0]["n_goal"]
if best_n_goal < N_AGENTS:
    print(f"\n[참고] 이번 스캔에서는 {N_AGENTS}척이 전부 성공한 판을 찾지 못했습니다. "
         f"찾은 것 중 가장 많이 성공한 판(최대 {best_n_goal}/{N_AGENTS})부터 보여드립니다.")

print(f"\n===== 상위 {TOP_N}개 판 =====")
for rank, ep_data in enumerate(top_episodes, start=1):
    print(f"[{rank}위] 목표도달 {ep_data['n_goal']}/{N_AGENTS} | score: {ep_data['score']:.2f} "
         f"(스캔 {ep_data['scan_index']}번째 판)")

# ===== 3단계: 재생 =====
for rank, ep_data in enumerate(top_episodes, start=1):
    print(f"\n[{rank}위 재생] 목표도달 {ep_data['n_goal']}/{N_AGENTS} | score: {ep_data['score']:.2f}")
    for i, result in enumerate(ep_data["agent_results"]):
        color = AGENT_COLOR_NAMES[i] if i < len(AGENT_COLOR_NAMES) else f"배{i}"
        mark = "✅ 성공" if result == "goal" else ("💥 충돌" if "collision" in result else "⏱ 시간초과")
        print(f"   - {color} 배: {mark} ({result})")

    for frame in ep_data["frames"]:
        env.positions = frame["positions"]
        env.headings = frame["headings"]
        env.done_flags = frame["done_flags"]
        env.step_count = frame["step_count"]
        env.render(pause=RENDER_PAUSE)
    time.sleep(1.0)

print("\n다 봤으면 창을 닫거나 Ctrl+C로 종료하세요.")
input("Enter를 누르면 프로그램을 종료합니다...")
env.close_render()
