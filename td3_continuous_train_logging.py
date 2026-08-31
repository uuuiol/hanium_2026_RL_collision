"""
td3_continuous_train_logging.py

boat_env.py에서 TD3(Twin Delayed DDPG)를 학습시킵니다.

TD3는 DDPG가 겪는 "Q값 과대평가 -> 정책 붕괴" 문제(ddpg_continuous_train_logging.py /
multi_ddpg_train_logging.py 실험에서 실제로 관찰됨)를 고치기 위해 2018년에 제안된 개선판입니다.
DDPG 대비 세 가지가 다릅니다.

1) Twin Critic (쌍둥이 크리틱): Q network를 2개(critic1, critic2) 따로 두고, 타겟을 계산할 때
   둘 중 "더 비관적인(작은)" 값을 씁니다. 한쪽이 어쩌다 과대평가해도 다른 쪽이 그걸 못 따라가게
   눌러주는 효과 -> DDPG 붕괴의 핵심 원인이었던 과대평가를 직접적으로 억제.
2) Delayed Policy Update (지연된 정책 업데이트): critic은 매 스텝 업데이트하지만, actor와
   타겟 네트워크들은 policy_delay(기본 2)스텝에 한 번만 업데이트합니다. critic이 좀 더 안정된
   뒤에 actor가 그걸 따라가게 해서, 덜 여문 critic을 actor가 성급하게 쫓아가다 같이 무너지는
   걸 막습니다.
3) Target Policy Smoothing (타겟 정책 스무딩): 타겟 Q를 계산할 때 쓰는 "다음 행동"에 작은
   클리핑된 노이즈를 더합니다. 특정 행동 하나에 뾰족하게 과최적화된 critic 값을 이용하지
   못하게 해서, 좁은 스파이크(급격한 Q값 튀는 지점)를 매끄럽게 만듭니다.

ReplayBuffer는 ddpg_continuous_train_logging.py 것을 그대로 가져다 씁니다 (알고리즘과 무관하게
전이(state, action, reward, next_state, bootstrap_mask)만 저장하면 되는 구조라 재사용 가능).
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from boat_env import BoatEnv
from ddpg_model import Actor, Critic
from ddpg_continuous_train_logging import ReplayBuffer, moving_average


class TD3Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # ===== TD3 하이퍼파라미터 =====
        self.gamma = 0.99
        self.tau = 0.005
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3

        # ===== TD3 고유 설정 (원 논문 기본값) =====
        self.policy_delay = 2          # actor/타겟은 critic보다 policy_delay배 느리게 업데이트
        self.target_noise_std = 0.2    # 타겟 정책 스무딩 노이즈 크기
        self.target_noise_clip = 0.5   # 그 노이즈를 이 범위로 클리핑

        self.actor = Actor(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.critic1 = Critic(state_size, action_size)
        self.critic2 = Critic(state_size, action_size)
        self.target_critic1 = Critic(state_size, action_size)
        self.target_critic2 = Critic(state_size, action_size)

        dummy_s = np.zeros([1, state_size], dtype=np.float32)
        dummy_a = np.zeros([1, action_size], dtype=np.float32)
        for net in (self.actor, self.target_actor):
            net(dummy_s)
        for net in (self.critic1, self.critic2, self.target_critic1, self.target_critic2):
            net(dummy_s, dummy_a)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.actor_optimizer = Adam(learning_rate=self.actor_lr)
        self.critic1_optimizer = Adam(learning_rate=self.critic_lr)
        self.critic2_optimizer = Adam(learning_rate=self.critic_lr)

        self.update_counter = 0

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

        # ===== 타겟 정책 스무딩: 다음 행동에 클리핑된 노이즈를 더해서 =====
        # critic이 특정 행동 하나에 뾰족하게 과최적화된 값을 이용하지 못하게 함
        raw_next_actions = self.target_actor(next_states)
        smoothing_noise = tf.clip_by_value(
            tf.random.normal(tf.shape(raw_next_actions), stddev=self.target_noise_std),
            -self.target_noise_clip, self.target_noise_clip)
        next_actions = tf.clip_by_value(raw_next_actions + smoothing_noise, -1.0, 1.0)

        # ===== Twin Critic: 둘 중 더 비관적인(작은) 값을 타겟으로 사용 -> 과대평가 억제 =====
        target_q1 = self.target_critic1(next_states, next_actions)
        target_q2 = self.target_critic2(next_states, next_actions)
        target_q = tf.minimum(target_q1, target_q2)
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

        avg_critic_loss = float(((critic1_loss + critic2_loss) / 2.0).numpy())

        # ===== Delayed Policy Update: critic보다 policy_delay배 느리게 actor/타겟 업데이트 =====
        self.update_counter += 1
        actor_loss_value = None
        if self.update_counter % self.policy_delay == 0:
            with tf.GradientTape() as tape:
                actions_pred = self.actor(states)
                actor_loss = -tf.reduce_mean(self.critic1(states, actions_pred))
            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            actor_loss_value = float(actor_loss.numpy())

            self._soft_update(self.target_actor, self.actor)
            self._soft_update(self.target_critic1, self.critic1)
            self._soft_update(self.target_critic2, self.critic2)

        return avg_critic_loss, actor_loss_value


if __name__ == "__main__":
    # ===== 실험 설정 (여기만 바꿔가며 실험하세요) =====
    RESULT_PREFIX = "td3_continuous"
    NUM_EPISODES = 300
    MAX_STEPS = 200
    # ======================================================

    env = BoatEnv(max_steps=MAX_STEPS)
    agent = TD3Agent(env.state_size, env.action_size)

    # ===== 리플레이 버퍼 / 학습 스케줄 설정 (DDPG와 동일) =====
    BUFFER_CAPACITY = 10_000
    BATCH_SIZE = 128
    WARMUP_STEPS = 1000
    rng = np.random.default_rng(0)
    buffer = ReplayBuffer(BUFFER_CAPACITY, env.state_size, env.action_size)

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
                if actor_loss is not None:
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
