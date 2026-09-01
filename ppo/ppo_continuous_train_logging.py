"""
ppo_continuous_train_logging.py

boat_env.py의 연속 행동(조향 각속도, 속도) 환경에 대해 PPO-continuous를 학습시키고,
에피소드별 결과를 CSV/그래프/궤적 이미지로 기록합니다. (TensorFlow/Keras)

핵심 구성요소 (지금까지 Deep SARSA/REINFORCE/DQN과 다른, PPO만의 정체성):
- Actor: 상태 -> 행동 평균(mu, tanh로 [-1,1] 제한), 표준편차(log_std)는 상태와 무관한 학습 파라미터
- Critic: 상태 -> 상태가치 V(s)  (Actor와 신경망을 공유하지 않고 별도 층 사용)
- GAE(Generalized Advantage Estimation)로 어드밴티지 계산
- 같은 rollout 데이터로 여러 epoch 동안, 클리핑된 목적함수로 반복 학습
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from common.boat_env import BoatEnv
from ppo.ppo_model import ActorCritic, gaussian_log_prob, gaussian_entropy


class PPOAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # PPO 하이퍼파라미터 (여기 값들을 바꿔가며 실험하면 됩니다)
        self.gamma = 0.99
        self.lam = 0.95          # GAE lambda
        self.clip_ratio = 0.2
        self.learning_rate = 0.0003
        self.train_epochs = 10   # 같은 rollout 데이터로 반복 학습하는 횟수 (PPO의 핵심 특징)
        self.batch_size = 64
        self.value_coef = 0.5
        self.entropy_coef = 0.01

        self.model = ActorCritic(state_size, action_size)
        # 최초 forward pass로 강제 build
        # (Deep SARSA/DQN 실험 때 겪었던 "trainable_variables가 빈 리스트" 버그를 미리 방지)
        self.model(np.zeros([1, state_size], dtype=np.float32))
        self.optimizer = Adam(learning_rate=self.learning_rate)

    def get_action(self, state):
        state_in = np.reshape(state, [1, -1]).astype(np.float32)
        mu, value = self.model(state_in)
        mu = mu.numpy()[0]
        log_std = self.model.log_std.numpy()
        std = np.exp(log_std)

        action = mu + std * np.random.randn(self.action_size)
        action = np.clip(action, -1.0, 1.0)

        log_prob = -0.5 * np.sum(((action - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
        return action.astype(np.float32), float(log_prob), float(value.numpy()[0, 0])

    def get_value(self, state):
        state_in = np.reshape(state, [1, -1]).astype(np.float32)
        _, value = self.model(state_in)
        return float(value.numpy()[0, 0])

    def compute_gae(self, rewards, values, dones, next_value):
        rewards = np.array(rewards, dtype=np.float32)
        values = np.array(list(values) + [next_value], dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self, states, actions, old_log_probs, advantages, returns):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        old_log_probs = np.array(old_log_probs, dtype=np.float32)
        advantages = np.array(advantages, dtype=np.float32)
        returns = np.array(returns, dtype=np.float32)

        # 어드밴티지 정규화 (학습 안정화)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)
        indices = np.arange(n)

        for _ in range(self.train_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                if len(batch_idx) == 0:
                    continue

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]

                with tf.GradientTape() as tape:
                    mu, value = self.model(b_states)
                    log_std = self.model.log_std
                    new_log_probs = gaussian_log_prob(b_actions, mu, log_std)

                    # ===== PPO의 핵심: 클리핑된 목적함수 =====
                    ratio = tf.exp(new_log_probs - b_old_log_probs)
                    surr1 = ratio * b_advantages
                    surr2 = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * b_advantages
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                    # ==========================================

                    value = tf.squeeze(value, axis=-1)
                    critic_loss = tf.reduce_mean(tf.square(b_returns - value))

                    entropy = gaussian_entropy(log_std)

                    loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                model_params = self.model.trainable_variables
                grads = tape.gradient(loss, model_params)
                self.optimizer.apply_gradients(zip(grads, model_params))


def moving_average(data, window=20):
    data = np.array(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


if __name__ == "__main__":
    # ===== 실험 설정 (여기만 바꿔가며 실험하세요) =====
    RESULT_PREFIX = "ppo_continuous"
    NUM_EPISODES = 300
    ROLLOUT_STEPS = 512     # 이만큼 경험을 모은 뒤 한 번씩 학습(업데이트)
    MAX_STEPS = 200         # 한 에피소드 최대 스텝 수
    # 참고: 이 스크립트는 파일 저장 전용 모드(matplotlib Agg)로 고정되어 있어서
    # 학습 도중 실시간으로 보는 기능은 넣지 않았습니다 (넣어도 창이 안 뜸).
    # 학습이 끝난 뒤 실시간으로 보고 싶으면 watch_trained_agent.py를 실행하세요.
    # ======================================================

    env = BoatEnv(max_steps=MAX_STEPS)
    agent = PPOAgent(env.state_size, env.action_size)

    episode_scores, episode_steps_list, episode_results, episode_numbers = [], [], [], []

    state = env.reset()
    trajectory_this_ep = [env.pos.copy()]
    episode_score = 0.0
    episode_step_count = 0
    episode_idx = 0
    last_trajectory = []

    rollout_states, rollout_actions, rollout_log_probs = [], [], []
    rollout_rewards, rollout_values, rollout_dones = [], [], []

    while episode_idx < NUM_EPISODES:
        for _ in range(ROLLOUT_STEPS):
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            rollout_states.append(state)
            rollout_actions.append(action)
            rollout_log_probs.append(log_prob)
            rollout_rewards.append(reward)
            rollout_values.append(value)
            rollout_dones.append(done)

            trajectory_this_ep.append(info["pos"])
            episode_score += reward
            episode_step_count += 1
            state = next_state

            if done:
                episode_idx += 1
                episode_scores.append(episode_score)
                episode_steps_list.append(episode_step_count)
                episode_results.append(info["result"])
                episode_numbers.append(episode_idx)

                if episode_idx % 5 == 0 or episode_idx <= 5:
                    print("episode: {:4d}/{} | score: {:7.2f} | steps: {:4d} | result: {}".format(
                        episode_idx, NUM_EPISODES, episode_score, episode_step_count, info["result"]))

                last_trajectory = trajectory_this_ep

                state = env.reset()
                trajectory_this_ep = [env.pos.copy()]
                episode_score = 0.0
                episode_step_count = 0

                if episode_idx >= NUM_EPISODES:
                    break

        # 모은 rollout으로 PPO 업데이트 (여러 에피소드 분량이 섞여있을 수 있음 - 정상)
        next_value = agent.get_value(state)
        advantages, returns = agent.compute_gae(rollout_rewards, rollout_values, rollout_dones, next_value)
        agent.update(rollout_states, rollout_actions, rollout_log_probs, advantages, returns)

        rollout_states, rollout_actions, rollout_log_probs = [], [], []
        rollout_rewards, rollout_values, rollout_dones = [], [], []

    # ===== CSV 저장 =====
    csv_path = f"{RESULT_PREFIX}_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "steps", "result"])
        for i in range(len(episode_numbers)):
            writer.writerow([episode_numbers[i], episode_scores[i], episode_steps_list[i], episode_results[i]])
    print(f"\n결과 CSV 저장 완료: {csv_path}")

    # ===== 학습곡선 그래프 =====
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    ma_score = moving_average(episode_scores, 20)
    axes[0].plot(episode_scores, alpha=0.3)
    axes[0].plot(range(len(episode_scores) - len(ma_score) + 1, len(episode_scores) + 1), ma_score, linewidth=2)
    axes[0].set_title(f"{RESULT_PREFIX} - Score per Episode")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Score")

    goal_flags = [1 if r == "goal" else 0 for r in episode_results]
    ma_goal = moving_average(goal_flags, 20) * 100
    axes[1].plot(range(len(goal_flags) - len(ma_goal) + 1, len(goal_flags) + 1), ma_goal, color="green")
    axes[1].set_title(f"{RESULT_PREFIX} - Goal-Reach Rate (20-episode moving avg)")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Goal Rate (%)")
    axes[1].set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(f"{RESULT_PREFIX}_training_result.png", dpi=120)
    print(f"그래프 저장 완료: {RESULT_PREFIX}_training_result.png")

    # ===== 마지막 에피소드 궤적 시각화 =====
    if last_trajectory:
        env.render_trajectory(last_trajectory, f"{RESULT_PREFIX}_last_trajectory.png")
        print(f"궤적 그래프 저장 완료: {RESULT_PREFIX}_last_trajectory.png")

    # ===== 학습된 모델 가중치 저장 (나중에 watch_trained_agent.py에서 불러다 씀) =====
    weights_path = f"{RESULT_PREFIX}_model_weights.weights.h5"
    agent.model.save_weights(weights_path)
    print(f"모델 가중치 저장 완료: {weights_path}")

    # ===== 요약 =====
    success_flags = [1 if r == "goal" else 0 for r in episode_results]
    collision_flags = [1 if r == "collision" else 0 for r in episode_results]
    print("\n===== 요약 =====")
    print(f"목표 도달률: {np.mean(success_flags) * 100:.1f}%")
    print(f"충돌률: {np.mean(collision_flags) * 100:.1f}%")
    print(f"평균 스텝: {np.mean(episode_steps_list):.2f}")
    print(f"평균 score: {np.mean(episode_scores):.2f}")
    if len(episode_scores) >= 20:
        print(f"초반 20개 평균 score: {np.mean(episode_scores[:20]):.2f}")
        print(f"후반 20개 평균 score: {np.mean(episode_scores[-20:]):.2f}")
