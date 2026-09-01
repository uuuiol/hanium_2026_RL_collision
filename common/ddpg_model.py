"""
ddpg_model.py

DDPG(Deep Deterministic Policy Gradient)용 Actor/Critic 신경망 정의.

PPO의 ActorCritic(하나의 신경망이 확률적 정책 mu+log_std와 가치 V(s)를 같이 냄)과 달리,
DDPG는 역할이 분리된 두 네트워크를 씁니다.
- Actor: 상태 -> 결정론적 행동(mu). tanh로 [-1,1] 범위로 제한.
- Critic: (상태, 행동) -> Q(s,a). "이 상태에서 이 행동을 하면 얼마나 좋은가"를 직접 평가.

둘 다 학습 안정화를 위해 "타겟 네트워크"(천천히 따라오는 사본)가 하나씩 더 필요합니다.
(TD 타겟을 계산할 때 지금 막 업데이트되는 네트워크를 그대로 쓰면 값이 출렁여서 발산하기 쉬움)
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


class Actor(Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(128, activation="relu")
        self.out = Dense(action_size, activation="tanh")

    def call(self, state):
        x = self.d1(state)
        x = self.d2(x)
        return self.out(x)


class Critic(Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(128, activation="relu")
        self.out = Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.d1(x)
        x = self.d2(x)
        return self.out(x)
