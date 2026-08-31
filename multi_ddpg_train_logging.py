"""
multi_ddpg_train_logging.py

MultiBoatEnv(여러 척의 배, 서로를 동적 장애물로 인식)에서
DDPG 정책 "하나"를 모든 에이전트가 공유하며 학습합니다 (parameter sharing, multi_ppo와 동일한 방식).

PPO와의 핵심 차이:
- PPO는 on-policy(방금 모은 경험으로만 학습하고 버림), DDPG는 off-policy(리플레이 버퍼에 경험을
  계속 쌓아두고 랜덤하게 뽑아서 반복 재사용).
- PPO는 확률적 정책(행동에 노이즈가 내장), DDPG는 결정론적 정책(mu 그대로 행동) + 탐험을 위해
  행동에 가우시안 노이즈를 수동으로 더함.
- PPO는 "월드 에피소드 하나가 끝난 뒤" 한 번에 업데이트하지만, DDPG는 매 스텝마다
  리플레이 버퍼에서 미니배치를 뽑아 즉시 업데이트합니다.

멀티 에이전트 처리 방식은 multi_ppo_train_logging.py와 동일합니다:
- 배 3척이 신경망 하나(Actor 하나 + Critic 하나)를 공유
- 각 배의 (상태, 행동, 보상, 다음상태, 종료여부) 전이를 전부 같은 리플레이 버퍼에 넣음
  (여기선 PPO처럼 궤적 단위로 분리할 필요가 없음 - DDPG는 애초에 전이 하나하나를 독립적으로 다룸)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_boat_env import MultiBoatEnv
from ddpg_continuous_train_logging import ReplayBuffer, DDPGAgent, moving_average


if __name__ == "__main__":
    # ===== 실험 설정 =====
    RESULT_PREFIX = "multi_ddpg_ep800"
    N_AGENTS = 3
    NUM_WORLD_EPISODES = 800
    MAX_STEPS = 200
    # ======================

    env = MultiBoatEnv(n_agents=N_AGENTS, max_steps=MAX_STEPS)
    agent = DDPGAgent(env.state_size, env.action_size)

    # ===== 리플레이 버퍼 / 학습 스케줄 설정 =====
    BUFFER_CAPACITY = 10_000
    BATCH_SIZE = 128
    WARMUP_STEPS = 1000     # 이 스텝까지는 actor 대신 완전 무작위 행동으로 버퍼를 다양하게 채움
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(BUFFER_CAPACITY, env.state_size, env.action_size)

    # ===== 탐험 노이즈 감쇠 (가우시안) =====
    # PPO의 학습률 감쇠와 같은 이유: 초반엔 크게 흔들며 탐험하고, 후반엔 이미 찾은 좋은 정책을
    # 살살 다듬도록 탐험을 줄임.
    NOISE_START = 0.3
    NOISE_END = 0.05

    # ===== 최고 성능 시점 저장 (multi_ppo와 동일한 방식) =====
    BEST_WINDOW = 20
    best_metric = -1e9
    best_weights_path = f"{RESULT_PREFIX}_actor_weights_best.weights.h5"

    world_scores, world_goal_counts, world_collision_counts, world_numbers = [], [], [], []
    world_colreg_rates = []

    csv_path = f"{RESULT_PREFIX}_results.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["world_episode", "avg_score", "goal_count", "collision_count",
                         "colreg_compliance_rate", "n_agents", "critic_loss", "actor_loss"])

    global_step = 0

    for world_ep in range(1, NUM_WORLD_EPISODES + 1):
        progress = (world_ep - 1) / max(NUM_WORLD_EPISODES - 1, 1)
        noise_sigma = NOISE_START + (NOISE_END - NOISE_START) * progress

        states = env.reset()
        episode_rewards = [[] for _ in range(N_AGENTS)]
        critic_losses_this_ep, actor_losses_this_ep = [], []

        while not env.all_done and env.step_count < MAX_STEPS:
            actions = []
            for i in range(N_AGENTS):
                if env.done_flags[i]:
                    actions.append([0.0, 0.0])
                    continue
                if global_step < WARMUP_STEPS:
                    a = rng.uniform(-1.0, 1.0, size=env.action_size).astype(np.float32)
                else:
                    a = agent.get_action(states[i], noise_sigma)
                actions.append(a)

            next_states, rewards, terminateds, truncateds, infos = env.step(actions)

            for i in range(N_AGENTS):
                if infos[i]["result"] == "already_done":
                    continue
                # 진짜 종료(목표도달/충돌)일 때만 bootstrap_mask=0. 타임아웃은 1로 둬서
                # "여기가 진짜 끝"이라고 잘못 학습하지 않게 함.
                bootstrap_mask = 0.0 if terminateds[i] else 1.0
                buffer.add(states[i], actions[i], rewards[i], next_states[i], bootstrap_mask)
                episode_rewards[i].append(rewards[i])
                global_step += 1

            states = next_states

            if buffer.size >= BATCH_SIZE and global_step >= WARMUP_STEPS:
                batch = buffer.sample(BATCH_SIZE, rng)
                critic_loss, actor_loss = agent.update(batch)
                critic_losses_this_ep.append(critic_loss)
                actor_losses_this_ep.append(actor_loss)

        # ===== 이번 월드 에피소드 결과 기록 =====
        results = [infos[i]["result"] for i in range(N_AGENTS)]
        n_goal = sum(1 for r in results if r == "goal")
        n_collision = sum(1 for r in results if "collision" in r)
        avg_score = np.mean([sum(episode_rewards[i]) for i in range(N_AGENTS)])
        colreg_rate = env.colreg_compliance_rate
        avg_critic_loss = float(np.mean(critic_losses_this_ep)) if critic_losses_this_ep else None
        avg_actor_loss = float(np.mean(actor_losses_this_ep)) if actor_losses_this_ep else None

        world_numbers.append(world_ep)
        world_scores.append(avg_score)
        world_goal_counts.append(n_goal)
        world_collision_counts.append(n_collision)
        world_colreg_rates.append(colreg_rate)

        csv_writer.writerow([world_ep, avg_score, n_goal, n_collision,
                            "" if colreg_rate is None else colreg_rate, N_AGENTS,
                            "" if avg_critic_loss is None else avg_critic_loss,
                            "" if avg_actor_loss is None else avg_actor_loss])
        csv_file.flush()

        # ===== 최고 성능 시점 저장 =====
        if len(world_goal_counts) >= BEST_WINDOW:
            recent_goal = np.mean(world_goal_counts[-BEST_WINDOW:])
            recent_score = np.mean(world_scores[-BEST_WINDOW:])
            metric = recent_goal * 1000 + recent_score
            if metric > best_metric:
                best_metric = metric
                agent.actor.save_weights(best_weights_path)
                print(f"  ★ 신기록! (최근 {BEST_WINDOW}ep 평균 목표도달 {recent_goal:.2f}, "
                     f"score {recent_score:.2f}) -> {best_weights_path} 저장")

        if world_ep % 10 == 0 or world_ep <= 5:
            colreg_str = "N/A" if colreg_rate is None else f"{colreg_rate*100:.0f}%"
            critic_str = "N/A" if avg_critic_loss is None else f"{avg_critic_loss:.4f}"
            print("world_ep {:4d}/{} | 에이전트평균 score: {:6.2f} | 목표도달 {}/{} | 충돌 {}/{} | "
                  "COLREG준수 {} | noise_sigma {:.3f} | buffer {} | critic_loss {}".format(
                world_ep, NUM_WORLD_EPISODES, avg_score, n_goal, N_AGENTS, n_collision, N_AGENTS,
                colreg_str, noise_sigma, buffer.size, critic_str))

    csv_file.close()
    print(f"\n결과 CSV 저장 완료: {csv_path}")

    # ===== 그래프 =====
    fig, axes = plt.subplots(4, 1, figsize=(10, 14))

    ma_score = moving_average(world_scores, 20)
    axes[0].plot(world_scores, alpha=0.3)
    axes[0].plot(range(len(world_scores) - len(ma_score) + 1, len(world_scores) + 1), ma_score, linewidth=2)
    axes[0].set_title(f"{RESULT_PREFIX} - Average Score per World-Episode")
    axes[0].set_xlabel("World Episode")
    axes[0].set_ylabel("Avg Score")

    goal_rate = [c / N_AGENTS * 100 for c in world_goal_counts]
    ma_goal = moving_average(goal_rate, 20)
    axes[1].plot(range(len(goal_rate) - len(ma_goal) + 1, len(goal_rate) + 1), ma_goal, color="green")
    axes[1].set_title(f"{RESULT_PREFIX} - Goal-Reach Rate (% of {N_AGENTS} agents, 20-ep moving avg)")
    axes[1].set_xlabel("World Episode")
    axes[1].set_ylabel("Goal Rate (%)")
    axes[1].set_ylim(-5, 105)

    collision_rate = [c / N_AGENTS * 100 for c in world_collision_counts]
    ma_collision = moving_average(collision_rate, 20)
    axes[2].plot(range(len(collision_rate) - len(ma_collision) + 1, len(collision_rate) + 1),
                ma_collision, color="red")
    axes[2].set_title(f"{RESULT_PREFIX} - Collision Rate (%, obstacle+agent combined, 20-ep moving avg)")
    axes[2].set_xlabel("World Episode")
    axes[2].set_ylabel("Collision Rate (%)")
    axes[2].set_ylim(-5, 105)

    colreg_rate_pct = [r * 100 for r in world_colreg_rates if r is not None]
    ma_colreg = moving_average(colreg_rate_pct, 20)
    if len(ma_colreg) > 0:
        axes[3].plot(range(len(colreg_rate_pct) - len(ma_colreg) + 1, len(colreg_rate_pct) + 1),
                    ma_colreg, color="darkorange")
    axes[3].set_title(f"{RESULT_PREFIX} - COLREG Compliance Rate (%, 20-ep moving avg, encounters only)")
    axes[3].set_xlabel("World Episode (조우 상황이 있었던 에피소드만)")
    axes[3].set_ylabel("Compliance Rate (%)")
    axes[3].set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(f"{RESULT_PREFIX}_training_result.png", dpi=120)
    print(f"그래프 저장 완료: {RESULT_PREFIX}_training_result.png")

    # ===== 모델 저장 =====
    weights_path = f"{RESULT_PREFIX}_actor_weights.weights.h5"
    agent.actor.save_weights(weights_path)
    print(f"모델 가중치 저장 완료(마지막 시점): {weights_path}")
    if best_metric > -1e9:
        print(f"최고 성능 시점 가중치: {best_weights_path}  <- 구경(watch)할 때는 이 파일을 쓰는 걸 권장!")

    # ===== 요약 =====
    print("\n===== 요약 =====")
    print(f"전체 목표도달률: {np.mean(goal_rate):.1f}%")
    print(f"전체 충돌률: {np.mean(collision_rate):.1f}%")
    if colreg_rate_pct:
        print(f"전체 COLREG 준수율(조우 발생 에피소드 기준): {np.mean(colreg_rate_pct):.1f}%")
    if len(world_scores) >= 20:
        print(f"초반 20개 월드에피소드 평균 목표도달률: {np.mean(goal_rate[:20]):.1f}%")
        print(f"후반 20개 월드에피소드 평균 목표도달률: {np.mean(goal_rate[-20:]):.1f}%")
        if len(colreg_rate_pct) >= 20:
            print(f"초반 20개 COLREG 준수율: {np.mean(colreg_rate_pct[:20]):.1f}%")
            print(f"후반 20개 COLREG 준수율: {np.mean(colreg_rate_pct[-20:]):.1f}%")
