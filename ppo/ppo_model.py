"""
ppo_model.py

PPO의 신경망(ActorCritic) 정의만 모아둔 파일입니다.
matplotlib을 전혀 쓰지 않기 때문에, 학습용 스크립트(ppo_continuous_train_logging.py)든
실시간 구경용 스크립트(watch_trained_agent.py)든 안전하게 가져다 쓸 수 있습니다.

(참고: 원래 이 내용이 ppo_continuous_train_logging.py 안에 같이 있었는데,
그 파일은 맨 위에서 matplotlib.use("Agg")를 선언해버려서 - 파일 저장 전용 모드 -
그 파일을 그대로 불러오면 실시간 창을 띄우는 다른 스크립트까지 덩달아
"화면에 안 그리는 모드"가 되어버리는 문제가 있었습니다. 그래서 분리했습니다.)
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense


class ActorCritic(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        # DDPG/TD3/SAC(128, relu)와 네트워크 용량을 맞추기 위해 hidden size/activation을 통일함.
        # mu_out은 행동을 [-1,1]로 제한해야 해서 relu가 아니라 tanh를 그대로 유지.
        self.shared1 = Dense(128, activation="relu")
        self.shared2 = Dense(128, activation="relu")
        self.mu_out = Dense(action_size, activation="tanh")

        self.critic1 = Dense(128, activation="relu")
        self.critic2 = Dense(128, activation="relu")
        self.value_out = Dense(1)

        # 행동 표준편차: 상태와 무관하게 학습되는 파라미터로 둠 (PPO-continuous의 흔한 관례)
        self.log_std = tf.Variable(initial_value=-0.5 * np.ones(action_size, dtype=np.float32),
                                    trainable=True, name="log_std")

    def call(self, x):
        h = self.shared1(x)
        h = self.shared2(h)
        mu = self.mu_out(h)

        v = self.critic1(x)
        v = self.critic2(v)
        value = self.value_out(v)

        return mu, value


def gaussian_log_prob(actions, mu, log_std):
    std = tf.exp(log_std)
    log_prob = -0.5 * (((actions - mu) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(log_prob, axis=-1)


def gaussian_entropy(log_std):
    return tf.reduce_sum(log_std + 0.5 * np.log(2 * np.pi * np.e))
