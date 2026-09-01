"""
vo_agent.py

VO(Velocity Obstacle) 기반 충돌회피 플래너입니다. APF와 마찬가지로 학습이 필요 없는
고전적(비-RL) baseline입니다.

아이디어(Fiorini & Shiller, 1998의 고전적 VO):
- 다른 배/장애물마다 "이 속도로 가면 언젠가 충돌한다"는 상대속도의 집합(Velocity Obstacle,
  콘 모양)이 있음: 상대위치 벡터를 중심으로 한 원뿔 안에 "내 속도 - 상대 속도"가 들어가면
  충돌 코스.
- 후보 속도(헤딩 x 속력을 격자로 촘촘히 샘플링)마다, 모든 근처 장애물/배에 대해 이 콘에
  걸리는지 검사해서 "안전한 후보"만 추림.
- 안전한 후보들 중 "목표 방향으로 최대속도(선호속도, preferred velocity)"에 가장 가까운 걸
  선택 -> 그 방향/속력을 원하는 헤딩/속도로 삼고, 비례제어기로 steer 행동을 만듦.

실무 구현에서 흔한 단순화를 그대로 따름:
- 정확한 기하 계산(콘 경계 교차) 대신, 후보 속도를 격자로 샘플링해서 각각 안전한지 검사
  (grid-sampling VO). 후보 수가 충분히 많으면 실질적으로 정확한 VO와 큰 차이 없음.
- 다른 배의 속도는 "직전 스텝 이동량 / dt"로 추정해서 씀 (이 배가 실제로 뭘 하려는지는
  모르니, "지금 이 속도로 계속 갈 것"이라고 가정하는 게 VO의 표준 가정).
"""
import numpy as np


class VOPlanner:
    def __init__(self, max_turn_rate, max_speed,
                 safety_margin=0.05, steer_gain=3.0,
                 n_heading_samples=36, speed_levels=(0.4, 0.6, 0.8, 1.0),
                 detection_range=1.5):
        self.max_turn_rate = max_turn_rate
        self.max_speed = max_speed
        self.safety_margin = safety_margin
        self.steer_gain = steer_gain
        self.n_heading_samples = n_heading_samples
        self.speed_levels = speed_levels
        self.detection_range = detection_range

    def _wrap(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _in_collision_cone(self, p_rel, v_rel, r_comb):
        dist = float(np.linalg.norm(p_rel))
        if dist <= r_comb:
            return True  # 이미 반경이 겹침
        half_angle = np.arcsin(np.clip(r_comb / dist, -1.0, 1.0))
        v_norm = float(np.linalg.norm(v_rel))
        if v_norm < 1e-6:
            return False  # 상대속도가 거의 0이면 실질적으로 접근하지 않음(실용적 근사)
        cos_angle = np.dot(p_rel, v_rel) / (dist * v_norm)
        angle_to_center = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return angle_to_center < half_angle

    def get_action(self, pos, heading, goal, own_radius, obstacles, obstacle_velocities, obstacle_radii):
        """
        obstacles: [obs_pos, ...] (고정 장애물 + 다른 배)
        obstacle_velocities: 대응하는 속도 벡터 (정적 장애물은 [0,0])
        obstacle_radii: 대응하는 반지름
        반환: [steer, speed] (둘 다 [-1,1])
        """
        pos = np.asarray(pos, dtype=np.float64)
        to_goal = np.asarray(goal, dtype=np.float64) - pos
        pref_heading = float(np.arctan2(to_goal[1], to_goal[0]))
        pref_speed = self.max_speed

        nearby = []
        for obs_pos, obs_vel, obs_r in zip(obstacles, obstacle_velocities, obstacle_radii):
            if np.linalg.norm(np.asarray(obs_pos) - pos) < self.detection_range:
                nearby.append((np.asarray(obs_pos, dtype=np.float64),
                               np.asarray(obs_vel, dtype=np.float64), obs_r))

        headings = np.linspace(-np.pi, np.pi, self.n_heading_samples, endpoint=False)
        best_cand = None
        best_score = np.inf

        for h in headings:
            for s_ratio in self.speed_levels:
                speed = s_ratio * self.max_speed
                v_cand = speed * np.array([np.cos(h), np.sin(h)])

                unsafe = False
                for obs_pos, obs_vel, obs_r in nearby:
                    p_rel = obs_pos - pos
                    v_rel = v_cand - obs_vel
                    r_comb = own_radius + obs_r + self.safety_margin
                    if self._in_collision_cone(p_rel, v_rel, r_comb):
                        unsafe = True
                        break
                if unsafe:
                    continue

                heading_diff = abs(self._wrap(h - pref_heading))
                turn_diff = abs(self._wrap(h - heading))  # 현재 헤딩에서 너무 급격히 안 틀도록 유도
                score = heading_diff + 0.3 * turn_diff + 0.1 * abs(speed - pref_speed)
                if score < best_score:
                    best_score = score
                    best_cand = (h, speed)

        if best_cand is None:
            # 안전한 후보가 하나도 없음(드묾) -> 최후의 수단으로 감속하며 현재 헤딩 유지
            desired_heading = heading
            desired_speed = 0.0
        else:
            desired_heading, desired_speed = best_cand

        heading_error = self._wrap(desired_heading - heading)
        steer = float(np.clip(heading_error * self.steer_gain / self.max_turn_rate, -1.0, 1.0))
        speed_action = float(np.clip(2.0 * (desired_speed / self.max_speed) - 1.0, -1.0, 1.0))
        return np.array([steer, speed_action], dtype=np.float32)
