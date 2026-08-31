"""
multi_ppo_train_logging.py

MultiBoatEnv(여러 척의 배, 서로를 동적 장애물로 인식)에서
PPO 정책 "하나"를 모든 에이전트가 공유하며 학습합니다 (parameter sharing).

핵심 아이디어:
- N척의 배가 있지만, 그 배들을 조종하는 신경망은 딱 하나뿐입니다.
- "N명의 학생이 각자 배운다"가 아니라 "한 명의 학생이 N대의 배를 동시에
  운전하며 경험을 N배 빨리 쌓는다"에 가깝습니다.
- 배끼리 부딪히면 둘 다 페널티를 받으므로, "다른 배를 피하는 것"도
  자연스럽게 정책에 녹아듭니다.

학습 방식:
- 한 "월드 에피소드"(모든 배가 끝나거나 최대 스텝에 도달할 때까지)를 다 진행한 뒤,
  각 에이전트별로 자신의 완결된 궤적에 대해 GAE를 따로 계산하고,
  그 결과를 전부 합쳐서 한 번에 PPO 업데이트합니다.
  (에이전트별 경험을 서로 뒤섞지 않고 각자 궤적 단위로 GAE를 계산하는 게 중요합니다 -
   안 그러면 A의 경험 끝에 B의 가치를 붙여서 잘못 계산하게 됩니다.)
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multi_boat_env import MultiBoatEnv
from ppo_continuous_train_logging import PPOAgent


def moving_average(data, window=20):
    data = np.array(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


if __name__ == "__main__":
    # ===== 실험 설정 =====
    RESULT_PREFIX = "multi_ppo_ep600"
    N_AGENTS = 3
    NUM_WORLD_EPISODES = 600   # "월드 에피소드" = 모든 배가 끝날 때까지의 한 판
    MAX_STEPS = 200
    # ======================

    env = MultiBoatEnv(n_agents=N_AGENTS, max_steps=MAX_STEPS)
    agent = PPOAgent(env.state_size, env.action_size)

    # ===== 개선 ③: 학습률 감쇠(Learning Rate Decay) 설정 =====
    # 학습 초반엔 과감하게(3e-4), 후반으로 갈수록 점점 조심스럽게(5e-5) 업데이트.
    # 고정 학습률로 끝까지 크게 업데이트하면, 후반에 잘 배운 정책이 다시 흔들려
    # 무너지는 현상(catastrophic forgetting)이 생길 수 있음 (PPO 표준 관행: lr annealing)
    LR_START = 3e-4
    LR_END = 5e-5

    # ===== 개선 ②: 최고 성능 시점 저장(Best Checkpoint) 설정 =====
    # 최근 BEST_WINDOW개 에피소드의 (목표도달수, score) 이동평균이 신기록이면 그 시점 가중치 저장.
    # 멀티에이전트 학습은 성능이 오르내릴 수 있어서, "마지막 시점"이 아니라
    # "가장 좋았던 시점"의 정책을 보존해야 실제 배포(실선박 탑재)에 쓸 수 있음.
    BEST_WINDOW = 20
    best_metric = -1e9
    best_weights_path = f"{RESULT_PREFIX}_model_weights_best.weights.h5"

    world_scores, world_goal_counts, world_collision_counts, world_numbers = [], [], [], []
    world_colreg_rates = []

    csv_path = f"{RESULT_PREFIX}_results.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["world_episode", "avg_score", "goal_count", "collision_count",
                         "colreg_compliance_rate", "n_agents"])

    for world_ep in range(1, NUM_WORLD_EPISODES + 1):
        # 개선 ③: 진행률에 따라 학습률을 선형으로 줄임 (매 에피소드 갱신)
        progress = (world_ep - 1) / max(NUM_WORLD_EPISODES - 1, 1)
        current_lr = LR_START + (LR_END - LR_START) * progress
        agent.optimizer.learning_rate.assign(current_lr)

        states = env.reset()

        # 에이전트별로 궤적을 따로 모아둠 (뒤섞지 않기 위해)
        agent_buffers = [{"states": [], "actions": [], "log_probs": [],
                          "rewards": [], "values": [], "dones": []} for _ in range(N_AGENTS)]

        while not env.all_done and env.step_count < MAX_STEPS:
            actions, log_probs_this_step, values_this_step = [], [], []
            for i in range(N_AGENTS):
                if env.done_flags[i]:
                    actions.append([0.0, 0.0])  # 이미 끝난 에이전트의 행동은 무시됨
                    log_probs_this_step.append(0.0)
                    values_this_step.append(0.0)
                    continue
                a, lp, v = agent.get_action(states[i])
                actions.append(a)
                log_probs_this_step.append(lp)
                values_this_step.append(v)

            next_states, rewards, terminateds, truncateds, infos = env.step(actions)

            for i in range(N_AGENTS):
                if infos[i]["result"] == "already_done":
                    continue
                done = terminateds[i] or truncateds[i]
                buf = agent_buffers[i]
                buf["states"].append(states[i])
                buf["actions"].append(actions[i])
                buf["log_probs"].append(log_probs_this_step[i])
                buf["rewards"].append(rewards[i])
                buf["values"].append(values_this_step[i])
                buf["dones"].append(done)

            states = next_states

        # ===== 월드 에피소드 종료: 에이전트별 GAE 계산 후 합쳐서 한 번에 업데이트 =====
        all_states, all_actions, all_log_probs = [], [], []
        all_advantages, all_returns = [], []
        for i in range(N_AGENTS):
            buf = agent_buffers[i]
            if len(buf["states"]) == 0:
                continue
            # 이 시점엔 모든 에이전트가 이미 종료(성공/충돌/타임아웃)된 상태이므로 next_value=0으로 안전
            advantages, returns = agent.compute_gae(buf["rewards"], buf["values"], buf["dones"], next_value=0.0)
            all_states.extend(buf["states"])
            all_actions.extend(buf["actions"])
            all_log_probs.extend(buf["log_probs"])
            all_advantages.extend(advantages)
            all_returns.extend(returns)

        agent.update(all_states, all_actions, all_log_probs, all_advantages, all_returns)

        # ===== 이번 월드 에피소드 결과 기록 =====
        results = [infos[i]["result"] for i in range(N_AGENTS)]
        n_goal = sum(1 for r in results if r == "goal")
        n_collision = sum(1 for r in results if "collision" in r)
        avg_score = np.mean([sum(agent_buffers[i]["rewards"]) for i in range(N_AGENTS)])
        colreg_rate = env.colreg_compliance_rate  # None이면 이번 에피소드엔 조우 상황 자체가 없었단 뜻

        world_numbers.append(world_ep)
        world_scores.append(avg_score)
        world_goal_counts.append(n_goal)
        world_collision_counts.append(n_collision)
        world_colreg_rates.append(colreg_rate)

        csv_writer.writerow([world_ep, avg_score, n_goal, n_collision,
                            "" if colreg_rate is None else colreg_rate, N_AGENTS])
        csv_file.flush()  # 중간에 프로그램이 끊겨도 여기까지 결과는 파일에 안전하게 남음

        # ===== 개선 ②: 최고 성능 시점 저장 =====
        # 최근 BEST_WINDOW개 이동평균으로 평가 (단일 에피소드의 운에 좌우되지 않도록).
        # 목표도달 수를 우선으로, 동률이면 score로 비교 (goal에 큰 가중치를 곱해 합성).
        if len(world_goal_counts) >= BEST_WINDOW:
            recent_goal = np.mean(world_goal_counts[-BEST_WINDOW:])
            recent_score = np.mean(world_scores[-BEST_WINDOW:])
            metric = recent_goal * 1000 + recent_score  # 목표도달 우선, score는 보조
            if metric > best_metric:
                best_metric = metric
                agent.model.save_weights(best_weights_path)
                print(f"  ★ 신기록! (최근 {BEST_WINDOW}ep 평균 목표도달 {recent_goal:.2f}, "
                     f"score {recent_score:.2f}) -> {best_weights_path} 저장")

        if world_ep % 10 == 0 or world_ep <= 5:
            colreg_str = "N/A" if colreg_rate is None else f"{colreg_rate*100:.0f}%"
            print("world_ep {:4d}/{} | 에이전트평균 score: {:6.2f} | 목표도달 {}/{} | 충돌 {}/{} | COLREG준수 {}".format(
                world_ep, NUM_WORLD_EPISODES, avg_score, n_goal, N_AGENTS, n_collision, N_AGENTS, colreg_str))

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
    weights_path = f"{RESULT_PREFIX}_model_weights.weights.h5"
    agent.model.save_weights(weights_path)
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
