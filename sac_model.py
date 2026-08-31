"""
sac_model.py

SAC(Soft Actor-Critic)용 Gaussian Actor 신경망 정의.

DDPG/TD3의 Actor(상태 -> 결정론적 행동 mu, tanh로 [-1,1] 제한)와 달리,
SAC의 Actor는 상태별로 정규분포 파라미터(mu, log_std)를 내고 그 분포에서 행동을 "샘플링"합니다
(확률적 정책). 그래서 DDPG/TD3처럼 탐험을 위해 행동에 외부 가우시안 노이즈를 수동으로 더할 필요가
없습니다 - 정책 자체가 확률적이라 탐험이 내장되어 있고, 학습 목적함수에 엔트로피 보너스까지 있어서
"너무 확신에 찬 정책으로 일찍 수렴하는 것"도 같이 억제됩니다.

reparameterization trick: 샘플링 과정(mu + std * eps, eps~N(0,1))이 미분 가능한 형태라
"행동을 샘플링했다"는 사실을 유지하면서도 actor 파라미터로 그래디언트가 그대로 흐릅니다.

tanh squashing: 샘플링한 값(pre_tanh)을 tanh로 [-1,1]에 밀어넣는데, 이러면 확률밀도도 같이
보정해줘야 합니다(변수변환 공식). log_prob 계산에서 그 보정항을 빼주는 부분이 그것입니다.

Critic은 ddpg_model.py의 Critic을 그대로 재사용합니다 (state, action) -> Q(s,a) 라는
역할 자체는 알고리즘과 무관하게 동일하기 때문입니다.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianActor(Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(128, activation="relu")
        self.mu_head = Dense(action_size)
        self.log_std_head = Dense(action_size)

    def call(self, state):
        x = self.d1(state)
        x = self.d2(x)
        mu = self.mu_head(x)
        log_std = tf.clip_by_value(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        """
        반환: (squashed 행동, 그 행동의 log_prob, tanh(mu)=평가/시연용 결정론적 행동)
        """
        mu, log_std = self(state)
        std = tf.exp(log_std)

        eps = tf.random.normal(tf.shape(mu))
        pre_tanh = mu + eps * std  # reparameterization trick
        action = tf.tanh(pre_tanh)

        # 정규분포 log_prob (squashing 전)
        log_prob = -0.5 * (tf.square((pre_tanh - mu) / (std + 1e-6)) + 2.0 * log_std + np.log(2.0 * np.pi))
        log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
        # tanh squashing 보정 (변수변환에 따른 야코비안 보정): log(1 - tanh(x)^2)
        log_prob -= tf.reduce_sum(tf.math.log(1.0 - tf.square(action) + 1e-6), axis=-1, keepdims=True)

        deterministic_action = tf.tanh(mu)
        return action, log_prob, deterministic_action
