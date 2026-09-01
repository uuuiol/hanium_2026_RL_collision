"""
apf_agent.py

APF(Artificial Potential Field, 인공potential장) 기반 충돌회피 플래너입니다.
학습이 필요 없는 고전적(비-RL) 방법으로, PPO/DDPG/TD3/SAC의 baseline 비교 대상으로 씁니다.

아이디어:
- 목표는 "당기는 힘"(인력, attractive force)을 만듦 - 항상 목표 방향으로 일정한 크기로 당김
  (거리에 비례하는 고전적 인력 공식은 목표 근처에서 진동하는 문제가 있어, 여기서는 방향만 쓰고
  크기는 고정한 "정규화된 인력"을 씀 - 실무에서 흔한 변형)
- 장애물/다른 배는 "미는 힘"(척력, repulsive force)을 만듦 - 영향범위(influence_radius) 안에
  들어오면 가까울수록 훨씬 세게 밀어냄 (표준 APF 척력 공식: 1/d - 1/d0 형태)
- 두 힘을 합친 방향을 "원하는 헤딩"으로 삼고, 비례제어기(P controller)로 steer 행동을 만듦
- 척력이 강할수록(=위험할수록) 속도를 줄여서 급조향에도 안정적으로 대응하게 함

한계(잘 알려진 APF의 고질병): 장애물들 사이에 갇히면 인력과 척력이 상쇄되어 멈춰버리는
"local minima" 문제가 있을 수 있음. 여기서는 최소속도(min_speed_ratio)를 둬서 완전정지는 피함.
"""
import numpy as np


class APFPlanner:
    def __init__(self, max_turn_rate, max_speed,
                 k_att=1.0, k_rep=0.6, influence_radius=1.0,
                 steer_gain=3.0, min_speed_ratio=0.3):
        self.max_turn_rate = max_turn_rate
        self.max_speed = max_speed
        self.k_att = k_att
        self.k_rep = k_rep
        self.influence_radius = influence_radius
        self.steer_gain = steer_gain
        self.min_speed_ratio = min_speed_ratio

    def _wrap(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_action(self, pos, heading, goal, obstacles):
        """
        pos, heading: 이 배의 현재 위치/헤딩
        goal: 이 배의 목표 위치
        obstacles: [(obs_pos, obs_radius), ...] 리스트 - 고정 장애물 + 다른 배(둘 다 동일하게 취급)
        반환: [steer, speed] (둘 다 [-1,1])
        """
        to_goal = np.asarray(goal, dtype=np.float64) - np.asarray(pos, dtype=np.float64)
        dist_goal = float(np.linalg.norm(to_goal)) + 1e-6
        f_att = self.k_att * to_goal / dist_goal

        f_rep = np.zeros(2, dtype=np.float64)
        for obs_pos, obs_radius in obstacles:
            delta = np.asarray(pos, dtype=np.float64) - np.asarray(obs_pos, dtype=np.float64)
            d = float(np.linalg.norm(delta))
            d_eff = d - obs_radius  # 장애물 "표면"까지의 거리
            if d_eff <= 1e-3:
                direction = delta / (d + 1e-6)
                f_rep += direction * self.k_rep * 1e3  # 거의 겹침 -> 최대 반발
            elif d_eff < self.influence_radius:
                mag = self.k_rep * (1.0 / d_eff - 1.0 / self.influence_radius) / (d_eff ** 2)
                f_rep += mag * delta / d

        f_total = f_att + f_rep
        rep_mag = float(np.linalg.norm(f_rep))

        if np.linalg.norm(f_total) < 1e-6:
            desired_heading = heading
        else:
            desired_heading = float(np.arctan2(f_total[1], f_total[0]))

        # 척력이 강할수록(=위험할수록) 감속 - 급조향 중 과속으로 인한 충돌 방지
        desired_speed = self.max_speed / (1.0 + rep_mag)
        desired_speed = max(desired_speed, self.max_speed * self.min_speed_ratio)

        heading_error = self._wrap(desired_heading - heading)
        steer = float(np.clip(heading_error * self.steer_gain / self.max_turn_rate, -1.0, 1.0))
        speed_action = float(np.clip(2.0 * (desired_speed / self.max_speed) - 1.0, -1.0, 1.0))
        return np.array([steer, speed_action], dtype=np.float32)
