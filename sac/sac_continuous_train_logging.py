"""
sac_continuous_train_logging.py

boat_env.py에서 SAC(Soft Actor-Critic)를 학습시킵니다.

DDPG/TD3와 SAC의 핵심 차이:
- DDPG/TD3는 결정론적 정책(mu 그대로 행동) + 탐험을 위해 행동에 가우시안 노이즈를 수동으로 더함.
  SAC는 확률적 정책(상태별 정규분포에서 행동을 샘플링)이라 탐험이 정책 자체에 내장됨.
- SAC의 목적함수는 "보상 + 엔트로피 보너스"를 최대화합니다 (entropy-regularized RL).
  엔트로피가 높을수록(=행동이 다양할수록) 추가 보너스를 줘서, 너무 일찍 하나의 행동에
  확신을 갖고 수렴해버리는 걸 막고 더 넓게 탐험하게 유도합니다.
- TD3처럼 twin critic(둘 중 작은 값 사용)을 그대로 씁니다 (Q 과대평가 억제, TD3와 동일 이유).
  다만 SAC는 별도의 target actor가 없습니다 - 정책이 이미 확률적이라 TD3의
  "타겟 정책 스무딩"과 비슷한 효과를 자체적으로 내기 때문입니다.
- 엔트로피 보너스의 크기(alpha, "온도")는 고정값을 쓰면 환경/학습 단계마다 최적값이 달라서
  튜닝이 까다롭습니다. 그래서 "목표 엔트로피"를 정해두고 alpha 자체도 경사하강으로 자동
  조정합니다(automatic temperature tuning, Haarnoja et al. 2018 SAC v2).

ReplayBuffer는 ddpg_continuous_train_logging.py 것을 그대로 가져다 씁니다.
"""
import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from common.boat_env import BoatEnv
from common.ddpg_model import Critic
from sac.sac_model import GaussianActor
from common.ddpg_continuous_train_logging import ReplayBuffer, moving_average


class SACAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # ===== SAC 하이퍼파라미터 =====
        # actor/critic/alpha 모두 같은 학습률을 쓰는 게 SAC 표준 관행입니다
        # (DDPG처럼 critic을 actor보다 빠르게 학습시켜야 할 이유가 없음 - twin critic +
        #  엔트로피 보너스 자체가 이미 학습을 안정화해줌).
        self.gamma = 0.99
        self.tau = 0.005
        self.actor_lr = 3e-4
        self.critic_lr = 3e-4
        self.alpha_lr = 3e-4

        self.actor = GaussianActor(state_size, action_size)
        self.critic1 = Critic(state_size, action_size)
        self.critic2 = Critic(state_size, action_size)
        self.target_critic1 = Critic(state_size, action_size)
        self.target_critic2 = Critic(state_size, action_size)

        dummy_s = np.zeros([1, state_size], dtype=np.float32)
        dummy_a = np.zeros([1, action_size], dtype=np.float32)
        self.actor(dummy_s)
        for net in (self.critic1, self.critic2, self.target_critic1, self.target_critic2):
            net(dummy_s, dummy_a)

        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic1_optimizer = Adam(learning_rate=self.critic_lr)
        self.critic2_optimizer = Adam(learning_rate=self.critic_lr)

        # ===== 자동 온도(alpha) 조정 =====
        # 목표 엔트로피 = -행동 차원 수 (SAC 논문에서 제안한 표준 휴리스틱).
        # log_alpha를 학습 파라미터로 두고 exp()를 취해 alpha를 얻음 (alpha > 0 보장 + 로그
        # 스케일이라 넓은 범위를 안정적으로 탐색 가능).
        self.target_entropy = -float(action_size)
        self.log_alpha = tf.Variable(np.log(0.2), dtype=tf.float32, trainable=True)
        self.alpha_optimizer = Adam(learning_rate=self.alpha_lr)

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def get_action(self, state, deterministic=False):
        state_in = np.reshape(state, [1, -1]).astype(np.float32)
        action, _, det_action = self.actor.sample(state_in)
        chosen = det_action if deterministic else action
        return chosen.numpy()[0].astype(np.float32)

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

        # ===== Critic 업데이트 =====
        # 타겟 Q에서 "엔트로피 보너스"까지 같이 부트스트랩 (- alpha * next_log_prob).
        # 다음 상태에서도 "다양한 선택지가 남아있는 것" 자체를 가치있게 침.
        next_actions, next_log_probs, _ = self.actor.sample(next_states)
        target_q1 = self.target_critic1(next_states, next_actions)
        target_q2 = self.target_critic2(next_states, next_actions)
        target_q = tf.minimum(target_q1, target_q2) - self.alpha * next_log_probs
        y = rewards + self.gamma * bootstrap_masks * target_q

        with tf.GradientTape(persistent=True) as tape:
            q1 = self.critic1(states, actions)
            q2 = self.critic2(states, actions)
            critic1_loss = tf.reduce_mean(tf.square(y - q1))
            critic2_loss = tf.reduce_mean(tf.square(y - q2))
        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        del tape
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))

        # ===== Actor 업데이트 =====
        # "Q가 커지는 방향"과 "엔트로피(다양성)가 커지는 방향"을 동시에 추구.
        with tf.GradientTape() as tape:
            new_actions, log_probs, _ = self.actor.sample(states)
            q1_new = self.critic1(states, new_actions)
            q2_new = self.critic2(states, new_actions)
            q_new = tf.minimum(q1_new, q2_new)
            actor_loss = tf.reduce_mean(self.alpha * log_probs - q_new)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # ===== 온도(alpha) 업데이트 =====
        # 실제 엔트로피(-log_prob)가 목표 엔트로피보다 낮으면(=정책이 너무 확신에 참) alpha를 키우고,
        # 반대로 너무 무작위로 행동하고 있으면 alpha를 줄임.
        with tf.GradientTape() as tape:
            alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy))
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)

        avg_critic_loss = float(((critic1_loss + critic2_loss) / 2.0).numpy())
        return avg_critic_loss, float(actor_loss.numpy()), float(self.alpha.numpy())


if __name__ == "__main__":
    # ===== 실험 설정 (여기만 바꿔가며 실험하세요) =====
    RESULT_PREFIX = "sac_continuous"
    NUM_EPISODES = 300
    MAX_STEPS = 200
    # ======================================================

    env = BoatEnv(max_steps=MAX_STEPS)
    agent = SACAgent(env.state_size, env.action_size)

    # ===== 리플레이 버퍼 / 학습 스케줄 설정 (DDPG/TD3와 동일) =====
    BUFFER_CAPACITY = 10_000
    BATCH_SIZE = 128
    WARMUP_STEPS = 1000     # 이 스텝까지는 actor 대신 완전 무작위 행동으로 버퍼를 다양하게 채움
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(BUFFER_CAPACITY, env.state_size, env.action_size)
    # SAC는 정책 자체가 확률적이라 DDPG/TD3처럼 별도의 노이즈 감쇠 스케줄이 필요 없습니다.

    episode_scores, episode_steps_list, episode_results, episode_numbers = [], [], [], []
    episode_critic_losses, episode_actor_losses, episode_alphas = [], [], []
    last_trajectory = []
    global_step = 0

    for episode_idx in range(1, NUM_EPISODES + 1):
        state = env.reset()
        trajectory_this_ep = [env.pos.copy()]
        episode_score = 0.0
        episode_step_count = 0
        done = False
        result = "running"
        critic_losses_this_ep, actor_losses_this_ep, alphas_this_ep = [], [], []

        while not done:
            if global_step < WARMUP_STEPS:
                action = rng.uniform(-1.0, 1.0, size=env.action_size).astype(np.float32)
            else:
                action = agent.get_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            result = info["result"]

            bootstrap_mask = 0.0 if terminated else 1.0
            buffer.add(state, action, reward, next_state, bootstrap_mask)

            trajectory_this_ep.append(info["pos"])
            episode_score += reward
            episode_step_count += 1
            global_step += 1
            state = next_state

            if buffer.size >= BATCH_SIZE and global_step >= WARMUP_STEPS:
                batch = buffer.sample(BATCH_SIZE, rng)
                critic_loss, actor_loss, alpha_val = agent.update(batch)
                critic_losses_this_ep.append(critic_loss)
                actor_losses_this_ep.append(actor_loss)
                alphas_this_ep.append(alpha_val)

        episode_scores.append(episode_score)
        episode_steps_list.append(episode_step_count)
        episode_results.append(result)
        episode_numbers.append(episode_idx)
        last_trajectory = trajectory_this_ep
        avg_critic_loss = float(np.mean(critic_losses_this_ep)) if critic_losses_this_ep else None
        avg_actor_loss = float(np.mean(actor_losses_this_ep)) if actor_losses_this_ep else None
        avg_alpha = float(np.mean(alphas_this_ep)) if alphas_this_ep else None
        episode_critic_losses.append(avg_critic_loss)
        episode_actor_losses.append(avg_actor_loss)
        episode_alphas.append(avg_alpha)

        if episode_idx % 5 == 0 or episode_idx <= 5:
            critic_str = "N/A" if avg_critic_loss is None else f"{avg_critic_loss:.4f}"
            alpha_str = "N/A" if avg_alpha is None else f"{avg_alpha:.4f}"
            print("episode: {:4d}/{} | score: {:7.2f} | steps: {:4d} | result: {} | "
                  "buffer: {} | critic_loss: {} | alpha: {}".format(
                episode_idx, NUM_EPISODES, episode_score, episode_step_count, result,
                buffer.size, critic_str, alpha_str))

    # ===== CSV 저장 =====
    csv_path = f"{RESULT_PREFIX}_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "steps", "result", "critic_loss", "actor_loss", "alpha"])
        for i in range(len(episode_numbers)):
            writer.writerow([episode_numbers[i], episode_scores[i], episode_steps_list[i], episode_results[i],
                            "" if episode_critic_losses[i] is None else episode_critic_losses[i],
                            "" if episode_actor_losses[i] is None else episode_actor_losses[i],
                            "" if episode_alphas[i] is None else episode_alphas[i]])
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

    if last_trajectory:
        env.render_trajectory(last_trajectory, f"{RESULT_PREFIX}_last_trajectory.png")
        print(f"궤적 그래프 저장 완료: {RESULT_PREFIX}_last_trajectory.png")

    weights_path = f"{RESULT_PREFIX}_actor_weights.weights.h5"
    agent.actor.save_weights(weights_path)
    print(f"모델 가중치 저장 완료: {weights_path}")

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
