"""
ddpg_continuous_train_logging.py

boat_env.py의 연속 행동(조향 각속도, 속도) 환경에 대해 DDPG를 학습시키고,
에피소드별 결과를 CSV/그래프/궤적 이미지로 기록합니다. (TensorFlow/Keras)

ppo_continuous_train_logging.py와 같은 환경(BoatEnv)을 쓰지만, 알고리즘 성격이 다릅니다.
- PPO: on-policy (방금 모은 경험으로만 학습하고 버림), 확률적 정책
- DDPG: off-policy (리플레이 버퍼에 경험을 계속 쌓아두고 랜덤 재사용), 결정론적 정책
  + 탐험을 위해 행동에 가우시안 노이즈를 수동으로 더함

DDPGAgent/ReplayBuffer는 multi_ddpg_train_logging.py(3척용)에서도 그대로 가져다 씁니다
(ppo_continuous_train_logging.py의 PPOAgent를 multi_ppo_train_logging.py가 가져다 쓰는 것과 동일한 구조).
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
from common.ddpg_model import Actor, Critic


class ReplayBuffer:
    def __init__(self, capacity, state_size, action_size):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_size), dtype=np.float32)
        self.actions = np.zeros((capacity, action_size), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_size), dtype=np.float32)
        # bootstrap_mask: 1.0이면 다음상태 가치를 이어붙임, 0.0이면 안 이어붙임.
        # "진짜로 끝남(목표도달/충돌)"일 때만 0으로 둠 - 타임아웃(시간초과)까지 0으로 두면
        # "여기서 에피소드가 원래 끝나야 한다"고 잘못 학습하는 고전적인 버그가 생김.
        self.bootstrap_masks = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, bootstrap_mask):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s2
        self.bootstrap_masks[self.ptr] = bootstrap_mask
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (self.states[idx], self.actions[idx], self.rewards[idx],
                self.next_states[idx], self.bootstrap_masks[idx])


class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # ===== DDPG 하이퍼파라미터 (여기 값들을 바꿔가며 실험하면 됩니다) =====
        self.gamma = 0.99
        self.tau = 0.005          # 타겟 네트워크 소프트 업데이트 비율 (작을수록 천천히 따라옴)
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3     # critic은 actor보다 좀 더 빨리 학습 (DDPG 표준 관행)

        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)

        dummy_s = np.zeros([1, state_size], dtype=np.float32)
        dummy_a = np.zeros([1, action_size], dtype=np.float32)
        self.actor(dummy_s)
        self.target_actor(dummy_s)
        self.critic(dummy_s, dummy_a)
        self.target_critic(dummy_s, dummy_a)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = Adam(learning_rate=self.critic_lr)

    def get_action(self, state, noise_sigma):
        state_in = np.reshape(state, [1, -1]).astype(np.float32)
        mu = self.actor(state_in).numpy()[0]
        noise = np.random.normal(0.0, noise_sigma, size=self.action_size)
        action = np.clip(mu + noise, -1.0, 1.0)
        return action.astype(np.float32)

    def _soft_update(self, target, source):
        new_weights = [self.tau * sw + (1 - self.tau) * tw
                       for tw, sw in zip(target.get_weights(), source.get_weights())]
        target.set_weights(new_weights)

    def update(self, batch):
        states, actions, rewards, next_states, bootstrap_masks = batch
        states = tf.constant(states, dtype=tf.float32)
        actions = tf.constant(actions, dtype=tf.float32)
        rewards = tf.constant(rewards, dtype=tf.float32)
        next_states = tf.constant(next_states, dtype=tf.float32)
        bootstrap_masks = tf.constant(bootstrap_masks, dtype=tf.float32)

        # ===== Critic 업데이트: TD 타겟(벨만 방정식)에 맞춰 Q(s,a) 회귀 =====
        next_actions = self.target_actor(next_states)
        target_q = self.target_critic(next_states, next_actions)
        y = rewards + self.gamma * bootstrap_masks * target_q

        with tf.GradientTape() as tape:
            q = self.critic(states, actions)
            critic_loss = tf.reduce_mean(tf.square(y - q))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        # ===== Actor 업데이트: 결정론적 정책 그래디언트 (Q가 커지는 방향으로 행동을 바꿈) =====
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(states, actions_pred))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)

        return float(critic_loss.numpy()), float(actor_loss.numpy())


def moving_average(data, window=20):
    data = np.array(data, dtype=float)
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


if __name__ == "__main__":
    # ===== 실험 설정 (여기만 바꿔가며 실험하세요) =====
    RESULT_PREFIX = "ddpg_continuous"
    NUM_EPISODES = 300
    MAX_STEPS = 200
    # ======================================================

    env = BoatEnv(max_steps=MAX_STEPS)
    agent = DDPGAgent(env.state_size, env.action_size)

    # ===== 리플레이 버퍼 / 학습 스케줄 설정 =====
    BUFFER_CAPACITY = 10_000
    BATCH_SIZE = 128
    WARMUP_STEPS = 1000     # 이 스텝까지는 actor 대신 완전 무작위 행동으로 버퍼를 다양하게 채움
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(BUFFER_CAPACITY, env.state_size, env.action_size)

    # ===== 탐험 노이즈 감쇠 (가우시안) =====
    NOISE_START = 0.3
    NOISE_END = 0.05

    episode_scores, episode_steps_list, episode_results, episode_numbers = [], [], [], []
    episode_critic_losses, episode_actor_losses = [], []
    last_trajectory = []
    global_step = 0

    for episode_idx in range(1, NUM_EPISODES + 1):
        progress = (episode_idx - 1) / max(NUM_EPISODES - 1, 1)
        noise_sigma = NOISE_START + (NOISE_END - NOISE_START) * progress

        state = env.reset()
        trajectory_this_ep = [env.pos.copy()]
        episode_score = 0.0
        episode_step_count = 0
        done = False
        result = "running"
        critic_losses_this_ep, actor_losses_this_ep = [], []

        while not done:
            if global_step < WARMUP_STEPS:
                action = rng.uniform(-1.0, 1.0, size=env.action_size).astype(np.float32)
            else:
                action = agent.get_action(state, noise_sigma)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            result = info["result"]

            # 진짜 종료(목표도달/충돌)일 때만 bootstrap_mask=0. 타임아웃은 1로 둬서
            # "여기가 진짜 끝"이라고 잘못 학습하지 않게 함.
            bootstrap_mask = 0.0 if terminated else 1.0
            buffer.add(state, action, reward, next_state, bootstrap_mask)

            trajectory_this_ep.append(info["pos"])
            episode_score += reward
            episode_step_count += 1
            global_step += 1
            state = next_state

            if buffer.size >= BATCH_SIZE and global_step >= WARMUP_STEPS:
                batch = buffer.sample(BATCH_SIZE, rng)
                critic_loss, actor_loss = agent.update(batch)
                critic_losses_this_ep.append(critic_loss)
                actor_losses_this_ep.append(actor_loss)

        episode_scores.append(episode_score)
        episode_steps_list.append(episode_step_count)
        episode_results.append(result)
        episode_numbers.append(episode_idx)
        last_trajectory = trajectory_this_ep
        avg_critic_loss = float(np.mean(critic_losses_this_ep)) if critic_losses_this_ep else None
        avg_actor_loss = float(np.mean(actor_losses_this_ep)) if actor_losses_this_ep else None
        episode_critic_losses.append(avg_critic_loss)
        episode_actor_losses.append(avg_actor_loss)

        if episode_idx % 5 == 0 or episode_idx <= 5:
            critic_str = "N/A" if avg_critic_loss is None else f"{avg_critic_loss:.4f}"
            print("episode: {:4d}/{} | score: {:7.2f} | steps: {:4d} | result: {} | "
                  "noise_sigma: {:.3f} | buffer: {} | critic_loss: {}".format(
                episode_idx, NUM_EPISODES, episode_score, episode_step_count, result,
                noise_sigma, buffer.size, critic_str))

    # ===== CSV 저장 =====
    csv_path = f"{RESULT_PREFIX}_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "steps", "result", "critic_loss", "actor_loss"])
        for i in range(len(episode_numbers)):
            writer.writerow([episode_numbers[i], episode_scores[i], episode_steps_list[i], episode_results[i],
                            "" if episode_critic_losses[i] is None else episode_critic_losses[i],
                            "" if episode_actor_losses[i] is None else episode_actor_losses[i]])
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

    # ===== 학습된 모델 가중치 저장 (나중에 watch 스크립트에서 불러다 씀) =====
    weights_path = f"{RESULT_PREFIX}_actor_weights.weights.h5"
    agent.actor.save_weights(weights_path)
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
