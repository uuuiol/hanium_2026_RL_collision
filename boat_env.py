"""
boat_env.py

그리드월드에서 벗어나, 연속 좌표(x, y) + 방위각(heading)으로 조향하는
아주 단순한 "보트" 내비게이션 환경입니다.

- 상태공간: 연속. 목표까지의 거리/상대방위, 장애물 3개까지의 거리/상대방위 (총 15차원)
- 행동공간: 연속 2차원 [조향(각속도), 속도], 둘 다 [-1, 1]로 정규화되어 입력됨
- 장애물: 고정 장애물 2개 + 왕복 이동하는 장애물 1개
  (기존 environment.py의 "고정 3개 장애물 + 목표" 구조를 연속공간으로 그대로 계승)
- 목표: 원형 목표 지점에 도달하면 종료 (+보상), 장애물 충돌/경계이탈 시에도 종료(-보상)

Gymnasium과 같은 (state, reward, terminated, truncated, info) 인터페이스를 따르되,
불필요한 의존성을 피하기 위해 gymnasium.Env를 상속하지는 않았습니다.
(numpy, matplotlib만 있으면 동작 — tkinter/이미지 에셋 불필요)
"""

import numpy as np


class BoatEnv:
    def __init__(self,
                 world_size=5.0,
                 max_steps=200,
                 dt=0.2,
                 max_turn_rate=1.5,       # rad/s, 최대 조향 각속도
                 max_speed=1.5,           # units/s, 최대 속도
                 goal_radius=0.3,
                 obstacle_radius=0.35,
                 collision_penalty=-10.0,
                 goal_reward=20.0,
                 step_penalty=-0.01,
                 progress_scale=2.0,
                 seed=None):
        self.world_size = world_size
        self.max_steps = max_steps
        self.dt = dt
        self.max_turn_rate = max_turn_rate
        self.max_speed = max_speed
        self.goal_radius = goal_radius
        self.obstacle_radius = obstacle_radius
        self.collision_penalty = collision_penalty
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.progress_scale = progress_scale

        self.action_size = 2   # [steer, speed]
        self.state_size = 15   # goal(3) + obstacle(4) * 3

        self.rng = np.random.default_rng(seed)

        # 고정 장애물 2개 + 왕복 이동 장애물 1개 (원본 3개 장애물 구조 계승)
        self.static_obstacles = [np.array([1.5, 3.5]), np.array([3.2, 1.2])]
        self.moving_obstacle_y = 2.5
        self.moving_obstacle_speed = 0.6

        self.goal_pos = np.array([world_size - 0.5, world_size - 0.5])

        self.pos = None
        self.heading = None
        self.moving_obstacle_x = None
        self.moving_obstacle_dir = None
        self.step_count = 0
        self._prev_dist_to_goal = None

        self.reset()

    # ---------- 내부 유틸 ----------
    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _bearing_to(self, target_pos):
        """에이전트가 바라보는 방향 기준, 목표/장애물이 상대적으로 어느 방향에 있는지"""
        delta = target_pos - self.pos
        dist = float(np.linalg.norm(delta))
        abs_angle = np.arctan2(delta[1], delta[0])
        bearing = self._wrap_angle(abs_angle - self.heading)
        return dist, bearing

    def _get_obstacle_positions(self):
        moving_pos = np.array([self.moving_obstacle_x, self.moving_obstacle_y])
        return self.static_obstacles + [moving_pos]

    def _get_state(self):
        dist_goal, bearing_goal = self._bearing_to(self.goal_pos)
        state = [dist_goal, np.sin(bearing_goal), np.cos(bearing_goal)]

        obstacle_positions = self._get_obstacle_positions()
        moving_flags = [0.0, 0.0, 1.0]  # 마지막 장애물만 움직임
        for obs_pos, moving_flag in zip(obstacle_positions, moving_flags):
            dist, bearing = self._bearing_to(obs_pos)
            state.extend([dist, np.sin(bearing), np.cos(bearing), moving_flag])

        return np.array(state, dtype=np.float32)

    # ---------- 외부 API ----------
    def reset(self):
        self.pos = np.array([0.5, 0.5], dtype=np.float32)
        self.heading = float(self.rng.uniform(-np.pi, np.pi))
        self.moving_obstacle_x = 2.5
        self.moving_obstacle_dir = 1
        self.step_count = 0
        self._prev_dist_to_goal = float(np.linalg.norm(self.goal_pos - self.pos))
        return self._get_state()

    def step(self, action):
        # action: [-1, 1] 범위의 2차원 연속값 -> 실제 물리 단위로 변환
        steer = float(np.clip(action[0], -1.0, 1.0)) * self.max_turn_rate
        speed_ratio = (float(np.clip(action[1], -1.0, 1.0)) + 1.0) / 2.0  # [0, 1]
        speed = speed_ratio * self.max_speed

        self.heading = self._wrap_angle(self.heading + steer * self.dt)
        self.pos = self.pos + np.array([np.cos(self.heading), np.sin(self.heading)],
                                       dtype=np.float32) * speed * self.dt
        # 경계는 즉시종료가 아니라 "벽"처럼 막습니다 (원본 tkinter 환경의 경계 처리 방식과 동일).
        # 시작 위치가 모서리 근처라 즉시종료로 두면 학습 신호가 생기기도 전에 에피소드가
        # 끝나버리는 문제가 있었습니다.
        self.pos = np.clip(self.pos, 0.0, self.world_size)

        # 움직이는 장애물 갱신 (x축 왕복)
        self.moving_obstacle_x += self.moving_obstacle_dir * self.moving_obstacle_speed * self.dt
        if self.moving_obstacle_x > self.world_size - 0.5:
            self.moving_obstacle_dir = -1
        elif self.moving_obstacle_x < 0.5:
            self.moving_obstacle_dir = 1

        self.step_count += 1

        dist_to_goal = float(np.linalg.norm(self.goal_pos - self.pos))
        reward = self.step_penalty
        reward += (self._prev_dist_to_goal - dist_to_goal) * self.progress_scale
        self._prev_dist_to_goal = dist_to_goal

        terminated = False
        result = "running"

        for obs_pos in self._get_obstacle_positions():
            if np.linalg.norm(obs_pos - self.pos) < self.obstacle_radius:
                reward += self.collision_penalty
                terminated = True
                result = "collision"
                break

        if not terminated and dist_to_goal < self.goal_radius:
            reward += self.goal_reward
            terminated = True
            result = "goal"

        truncated = False
        if not terminated and self.step_count >= self.max_steps:
            truncated = True
            result = "timeout"

        info = {"result": result, "pos": self.pos.copy(), "heading": self.heading}
        return self._get_state(), reward, terminated, truncated, info

    # ---------- 실시간 시각화 (학습된 정책을 눈으로 지켜볼 때 사용) ----------
    def render(self, pause=0.05, trail=None):
        """
        매 스텝 호출하면, 보트가 실제로 움직이는 과정을 창에 실시간으로 그려줍니다.
        (학습 도중 매 스텝 호출하면 매우 느려지므로, 보통은 학습이 끝난 뒤
         watch_trained_agent.py처럼 완성된 정책을 "구경"할 때만 사용합니다.)
        """
        import matplotlib.pyplot as plt

        if not hasattr(self, "_fig") or self._fig is None:
            plt.ion()  # 대화형 모드: 창을 띄운 채로 내용만 계속 갱신
            self._fig, self._ax = plt.subplots(figsize=(6, 6))

        ax = self._ax
        ax.clear()

        for obs_pos in self.static_obstacles:
            ax.add_patch(plt.Circle(obs_pos, self.obstacle_radius, color="steelblue", alpha=0.6))
        ax.add_patch(plt.Circle([self.moving_obstacle_x, self.moving_obstacle_y],
                                self.obstacle_radius, color="orange", alpha=0.6))
        ax.add_patch(plt.Circle(self.goal_pos, self.goal_radius, color="green", alpha=0.4))

        if trail:
            trail_arr = np.array(trail)
            ax.plot(trail_arr[:, 0], trail_arr[:, 1], "r-", alpha=0.4, linewidth=1)

        # 보트 본체 + 방향(heading) 화살표 -- 여기가 "각도"를 눈으로 보여주는 부분
        ax.plot(self.pos[0], self.pos[1], "ro", markersize=10)
        arrow_len = 0.4
        ax.arrow(self.pos[0], self.pos[1],
                 arrow_len * np.cos(self.heading), arrow_len * np.sin(self.heading),
                 head_width=0.15, color="darkred", length_includes_head=True)

        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect("equal")
        ax.set_title(f"step {self.step_count}")

        plt.pause(pause)

    def close_render(self):
        import matplotlib.pyplot as plt
        if hasattr(self, "_fig") and self._fig is not None:
            plt.close(self._fig)
            self._fig = None

    # ---------- 시각화 (학습 후 궤적 확인용, 파일로 저장) ----------
    def render_trajectory(self, trajectory, save_path):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))

        for obs_pos in self.static_obstacles:
            ax.add_patch(plt.Circle(obs_pos, self.obstacle_radius, color="steelblue", alpha=0.6))
        ax.add_patch(plt.Circle([self.moving_obstacle_x, self.moving_obstacle_y],
                                self.obstacle_radius, color="orange", alpha=0.6,
                                label="moving obstacle (final pos)"))
        ax.add_patch(plt.Circle(self.goal_pos, self.goal_radius, color="green", alpha=0.4, label="goal"))

        traj = np.array(trajectory)
        ax.plot(traj[:, 0], traj[:, 1], "r-", linewidth=1.5, label="trajectory")
        ax.plot(traj[0, 0], traj[0, 1], "ko", label="start")
        ax.plot(traj[-1, 0], traj[-1, 1], "kx", markersize=10, label="end")

        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title("BoatEnv Episode Trajectory")
        plt.savefig(save_path, dpi=120)
        plt.close(fig)
