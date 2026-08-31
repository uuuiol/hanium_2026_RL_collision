"""
watch_trained_agent.py

ppo_continuous_train_logging.py로 학습을 끝까지 마치면 생성되는
"{RESULT_PREFIX}_model_weights.*" 파일들을 불러와서,
학습된 보트가 실제로 어떻게 움직이는지 화면에 실시간으로 보여줍니다.

먼저 ppo_continuous_train_logging.py를 끝까지 실행해서 가중치 파일이
생성된 뒤에 이 스크립트를 실행하세요.

(주의) 학습 스크립트와 달리 이 파일은 matplotlib.use("Agg")를 쓰지 않습니다.
그래야 실제 화면 창이 뜹니다.
"""
import time
import numpy as np

from boat_env import BoatEnv
from ppo_model import ActorCritic

# ===== 설정 =====
WEIGHTS_PREFIX = "ppo_continuous_model_weights.weights.h5"  # 학습 스크립트의 RESULT_PREFIX와 맞춰주세요
NUM_EPISODES_TO_WATCH = 5
RENDER_PAUSE = 0.03    # 프레임 사이 대기 시간(초). 숫자를 키우면 더 천천히, 줄이면 더 빠르게 보입니다
MAX_STEPS = 200
# ================

env = BoatEnv(max_steps=MAX_STEPS)

model = ActorCritic(env.state_size, env.action_size)
model(np.zeros([1, env.state_size], dtype=np.float32))  # 강제 build (가중치를 불러오기 전에 필요)
model.load_weights(WEIGHTS_PREFIX)
print(f"모델 가중치 불러오기 완료: {WEIGHTS_PREFIX}")

for ep in range(NUM_EPISODES_TO_WATCH):
    state = env.reset()
    trail = [env.pos.copy()]
    done = False
    env.render(pause=RENDER_PAUSE, trail=trail)

    while not done:
        state_in = np.reshape(state, [1, -1]).astype(np.float32)
        mu, _ = model(state_in)
        # 구경할 때는 탐험용 무작위 노이즈를 섞지 않고,
        # 신경망이 "가장 확신하는" 행동(mu)을 그대로 사용합니다.
        action = np.clip(mu.numpy()[0], -1.0, 1.0)

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        trail.append(env.pos.copy())

        env.render(pause=RENDER_PAUSE, trail=trail)

    print(f"episode {ep + 1}/{NUM_EPISODES_TO_WATCH} | result: {info['result']} | steps: {env.step_count}")
    time.sleep(0.5)  # 다음 에피소드 시작 전 잠깐 정지 (결과를 눈으로 확인할 시간)

print("\n다 봤으면 창을 닫거나 Ctrl+C로 종료하세요.")
input("Enter를 누르면 프로그램을 종료합니다...")
env.close_render()
