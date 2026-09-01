"""
multi_boat_env.py

여러 척(N_AGENTS)의 배가 서로를 동적 장애물로 인식하며 학습하는 멀티 에이전트 환경입니다.
여기에 COLREG(국제해상충돌예방규칙)의 핵심 3개 조우 상황(정면마주침/추월/교차)에 대한
판정 로직과 보상 유도를 추가했습니다.

COLREG 처리 방식 요약:
- 두 배가 서로 "조우 범위(encounter_range)" 안에 들어오면, 방위각(bearing)과 진행방향 차이로
  상황을 head_on(정면마주침) / give_way(내가 피해야 함) / stand_on(내가 우선통항) 중 하나로 판정
- 규칙을 강제하지 않고, "규칙대로 행동하면 작은 보너스, 어기면 작은 페널티"만 줘서 유도
  (실제로 그게 유리한지는 에이전트가 스스로 학습)
- 상태(observation)에도 "지금 이 상대와 나 사이에서 내 역할이 뭔지"를 포함시켜서
  에이전트가 이 정보를 보고 판단할 수 있게 함
- colreg_enabled=False로 두면 COLREG 없이 이전 버전과 동일하게 동작 (비교 실험용)
"""
import numpy as np


class MultiBoatEnv:
    def __init__(self,
                 n_agents=3,
                 world_size=5.0,
                 max_steps=200,
                 dt=0.2,
                 max_turn_rate=1.5,
                 max_speed=1.5,
                 goal_radius=0.3,
                 obstacle_radius=0.35,
                 agent_radius=0.25,
                 collision_penalty=-20.0,
                 # ===== 안전거리 관련: DCPA/TCPA 기반 충돌위험도(CR) 계산에 씀 =====
                 # COLREG 제8조 "충분히 여유 있는 거리(safe distance)를 두고 피항" 반영.
                 safety_distance=0.7,
                 goal_reward=20.0,
                 step_penalty=-0.01,
                 progress_scale=2.0,
                 seed=None,
                 # ===== COLREG 관련 설정 =====
                 colreg_enabled=True,
                 encounter_range=1.2,        # 이 거리 안에 다른 배가 있으면 "조우 상황"으로 판단
                 head_on_cone_deg=20.0,       # 상대가 이 각도 이내로 정면에 있으면 head-on 후보
                 same_course_cone_deg=30.0,   # 진행방향 차이가 이 각도 이내면 "같은 방향"으로 간주
                 overtake_cone_deg=45.0,      # 상대가 이 각도 이내로 앞/뒤에 있으면 추월상황 후보
                 colreg_steer_threshold=0.1,  # 이 이상 조향해야 "우현으로 틀었다/직진했다"로 인정
                 # ===== DCPA/TCPA 기반 충돌위험도(CR, Collision Risk) 가중치 =====
                 # 참고: Mou et al. 2010 / arXiv:2211.01004 / fmars.2025.1756233 / Oxford JCDE 2021, qwab029
                 w_CR=2.0,        # Near-Miss(CR) 페널티 최대 크기 (목표보상 20의 10%)
                 w_col=1.0,       # COLREG 보상 최대 크기 (CR=1일 때 ±1.0)
                 w_heading=0.3,   # 목표방향 헤딩 보상 최대 크기
                 cr_c1=None,      # None이면 safety_distance에서 CR=0.1이 되도록 log(0.1)/safety_distance로 자동 설정
                 cr_c2=1.5,       # TCPA>=0(접근 중)일 때 가중치
                 cr_c3=20.0,      # TCPA<0(이미 지나침)일 때 가중치 - 크게 둬서 위험도를 빠르게 감쇠
                 ):
        self.n_agents = n_agents
        self.world_size = world_size
        self.max_steps = max_steps
        self.dt = dt
        self.max_turn_rate = max_turn_rate
        self.max_speed = max_speed
        self.goal_radius = goal_radius
        self.obstacle_radius = obstacle_radius
        self.agent_radius = agent_radius
        self.collision_penalty = collision_penalty
        self.safety_distance = safety_distance
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.progress_scale = progress_scale

        self.colreg_enabled = colreg_enabled
        self.encounter_range = encounter_range
        self.head_on_cone = np.radians(head_on_cone_deg)
        self.same_course_cone = np.radians(same_course_cone_deg)
        self.overtake_cone = np.radians(overtake_cone_deg)
        self.colreg_steer_threshold = colreg_steer_threshold

        self.w_CR = w_CR
        self.w_col = w_col
        self.w_heading = w_heading
        self.cr_c1 = cr_c1 if cr_c1 is not None else np.log(0.1) / self.safety_distance
        self.cr_c2 = cr_c2
        self.cr_c3 = cr_c3

        self.action_size = 2  # [steer, speed]
        # 상태크기 = 목표(3) + 고정장애물 2개*4 + (다른 에이전트 수)*4  (구조는 이전과 동일,
        # 다만 다른 에이전트 블록의 마지막 값이 "동적여부(1.0 고정)" 대신 "COLREG 역할 코드"로 바뀜)
        self.state_size = 3 + 2 * 4 + (n_agents - 1) * 4

        self.rng = np.random.default_rng(seed)

        # 장애물 위치: 세 에이전트의 직선경로 모두로부터 비슷한 여유(agent0/1=1.27, agent2=1.80)를
        # 갖도록 재배치함. 원래는 agent0(빨강)의 경로만 장애물에 0.4~0.6 거리로 바짝 붙어있어서
        # 구조적으로 훨씬 불리했음 (충돌 판정 반경 obstacle_radius=0.35 대비 여유가 0.07밖에 없었음).
        self.static_obstacles = [np.array([2.5, 0.7]), np.array([2.5, 4.3])]

        default_starts_goals = [
            (np.array([0.5, 0.5]), np.array([4.5, 4.5])),
            (np.array([4.5, 0.5]), np.array([0.5, 4.5])),
            (np.array([0.5, 2.5]), np.array([4.5, 2.5])),
            (np.array([4.5, 4.5]), np.array([0.5, 0.5])),
            (np.array([2.5, 0.3]), np.array([2.5, 4.7])),
        ]
        if n_agents > len(default_starts_goals):
            raise ValueError(f"n_agents는 최대 {len(default_starts_goals)}까지 지원합니다.")
        self.starts = [sg[0] for sg in default_starts_goals[:n_agents]]
        self.goals = [sg[1] for sg in default_starts_goals[:n_agents]]

        self.positions = None
        self.headings = None
        self.done_flags = None
        self.step_count = 0
        self._prev_dist_to_goal = None

        # COLREG 준수율 집계용 (에피소드 단위로 reset()에서 초기화)
        self.colreg_checks = 0
        self.colreg_compliant = 0

        # CR(충돌위험도)/COLREG 보상 로깅용 (에피소드 단위로 reset()에서 초기화)
        self.cr_history = []
        self.colreg_reward_history = []

        self.reset()

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _bearing_angle(self, delta):
        return float(np.arctan2(delta[1], delta[0]))

    def _bearing_from(self, agent_idx, target_pos):
        delta = target_pos - self.positions[agent_idx]
        dist = float(np.linalg.norm(delta))
        abs_angle = np.arctan2(delta[1], delta[0])
        bearing = self._wrap_angle(abs_angle - self.headings[agent_idx])
        return dist, bearing

    # ---------- COLREG 상황 판정 ----------
    def _classify_colreg(self, i, j):
        """
        에이전트 i가 에이전트 j와의 관계에서 COLREG상 어떤 역할인지 판정.
        반환: "head_on" / "give_way" / "stand_on" / None(조우범위 밖 = 해당없음)

        bearing 부호 규약: 양수(+) = 상대가 내 좌현(port)에 있음, 음수(-) = 상대가 내 우현(starboard)에 있음
        (heading은 반시계방향이 양수인 표준 수학 각도 규약을 쓰기 때문)
        """
        dist = float(np.linalg.norm(self.positions[j] - self.positions[i]))
        if dist > self.encounter_range:
            return None

        _, bearing_j = self._bearing_from(i, self.positions[j])
        heading_diff = self._wrap_angle(self.headings[j] - self.headings[i])

        abs_bearing = abs(bearing_j)
        abs_heading_diff = abs(heading_diff)

        # 1) 정면 마주침(head-on): 상대가 거의 정면 + 서로 거의 반대 방향
        if abs_bearing < self.head_on_cone and abs_heading_diff > (np.pi - self.head_on_cone):
            return "head_on"

        # 2) 내가 추월하는 입장: 거의 같은 방향 + 상대가 내 앞쪽
        if abs_heading_diff < self.same_course_cone and abs_bearing < self.overtake_cone:
            return "give_way"

        # 3) 내가 추월당하는 입장: 거의 같은 방향 + 상대가 내 뒤쪽
        if abs_heading_diff < self.same_course_cone and abs_bearing > (np.pi - self.overtake_cone):
            return "stand_on"

        # 4) 교차(crossing): 상대가 내 우현이면 내가 give-way, 좌현이면 stand-on
        return "give_way" if bearing_j < 0 else "stand_on"

    def _colreg_role_code(self, role):
        return {"head_on": 2.0, "give_way": -1.0, "stand_on": 1.0, None: 0.0}[role]

    def _colreg_delta(self, role, steer):
        """
        COLREG 규칙 준수 정도를 [-1, +1]로 반환 (양수=준수, 음수=위반, 0=중립).
        _compute_cr()로 구한 위험도(CR)와 곱해져서 실제 보상 크기가 됨 - 위험할수록(CR↑)
        규칙 준수/위반에 대한 신호가 커지고, 위험하지 않으면(CR≈0) 거의 무시됨.
        """
        if role in ("head_on", "give_way"):
            if steer < -self.colreg_steer_threshold:
                return 1.0     # 우현 변침 (COLREGs Rule 14/15)
            elif steer > self.colreg_steer_threshold:
                return -1.0    # 좌현 변침 (위반)
            return 0.0          # 직진 (중립)
        elif role == "stand_on":
            if abs(steer) < self.colreg_steer_threshold:
                return 1.0      # 침로 유지 (Rule 17)
            return -0.5          # 불필요한 변침 (약한 페널티)
        return 0.0

    # ---------- DCPA/TCPA/CR (충돌위험도) ----------
    def _compute_dcpa_tcpa(self, pos_own, vel_own, pos_target, vel_target):
        """
        DCPA(최근접거리)/TCPA(최근접시간)를 상대위치/상대속도 벡터로 계산.
        TCPA > 0: 아직 최근접점에 도달 안 함(접근 중). TCPA < 0: 이미 최근접점을 지남(멀어지는 중).
        상대속도가 거의 0이면(둘 다 같은 방향/속도로 나란히 움직임) DCPA=현재거리, TCPA=+inf로 처리.
        """
        r_vec = pos_target - pos_own
        v_vec = vel_target - vel_own
        v_norm = float(np.linalg.norm(v_vec))

        if v_norm < 1e-6:
            return float(np.linalg.norm(r_vec)), float("inf")

        cross = r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0]  # 2D 외적의 크기(스칼라)
        dcpa = abs(cross) / v_norm
        tcpa = -float(np.dot(r_vec, v_vec)) / (v_norm ** 2)
        return dcpa, tcpa

    def _compute_cr(self, dcpa, tcpa, d_euclidean):
        """
        DCPA/TCPA + 현재 유클리드거리로 충돌위험도 CR을 [0,1]로 계산.
        참고: Mou et al. 2010(DCPA/TCPA 지수함수 모델), arXiv:2211.01004(CR_CPA 공식/c1,c2,c3),
        fmars.2025.1756233(COLREG 보상 x 위험도 연동), Oxford JCDE 2021 qwab029(DCPA/TCPA 검증).
        """
        weight = self.cr_c2 if tcpa >= 0 else self.cr_c3
        tcpa_term = weight * abs(tcpa) if np.isfinite(tcpa) else 0.0
        exponent_cpa = np.clip(self.cr_c1 * (dcpa + tcpa_term), -500.0, 0.0)
        cr_cpa = float(np.exp(exponent_cpa))

        exponent_ed = np.clip(self.cr_c1 * d_euclidean, -500.0, 0.0)
        cr_ed = float(np.exp(exponent_ed))

        return max(cr_cpa, cr_ed)

    # ---------- 상태 ----------
    def _get_state(self, agent_idx):
        dist_goal, bearing_goal = self._bearing_from(agent_idx, self.goals[agent_idx])
        state = [dist_goal, np.sin(bearing_goal), np.cos(bearing_goal)]

        for obs_pos in self.static_obstacles:
            dist, bearing = self._bearing_from(agent_idx, obs_pos)
            state.extend([dist, np.sin(bearing), np.cos(bearing), 0.0])

        for j in range(self.n_agents):
            if j == agent_idx:
                continue
            dist, bearing = self._bearing_from(agent_idx, self.positions[j])
            if self.colreg_enabled:
                role = self._classify_colreg(agent_idx, j)
                role_code = self._colreg_role_code(role)
            else:
                role_code = 1.0  # COLREG 끄면 이전처럼 "동적 장애물이다"만 표시
            state.extend([dist, np.sin(bearing), np.cos(bearing), role_code])

        return np.array(state, dtype=np.float32)

    def _get_all_states(self):
        return [self._get_state(i) for i in range(self.n_agents)]

    def _sample_random_start_goal(self, existing_starts, wall_margin=0.3, obstacle_margin=0.5,
                                   min_agent_sep=0.8, min_travel=2.0, max_tries=200):
        """
        월드 전체 범위에서 무작위 시작/목표 좌표를 뽑음 (일반화 테스트용).
        - 벽에서 wall_margin 이상 떨어짐
        - 고정 장애물에서 obstacle_margin 이상 떨어짐
        - 시작점은 다른 배들의 시작점과 min_agent_sep 이상 떨어짐 (스폰하자마자 충돌 방지)
        - 목표는 시작점에서 min_travel 이상 떨어짐 (너무 쉬운 판 방지)
        """
        for _ in range(max_tries):
            start = self.rng.uniform(wall_margin, self.world_size - wall_margin, size=2)
            if any(np.linalg.norm(start - obs) < self.obstacle_radius + obstacle_margin
                   for obs in self.static_obstacles):
                continue
            if any(np.linalg.norm(start - p) < min_agent_sep for p in existing_starts):
                continue
            break

        for _ in range(max_tries):
            goal = self.rng.uniform(wall_margin, self.world_size - wall_margin, size=2)
            if np.linalg.norm(goal - start) < min_travel:
                continue
            if any(np.linalg.norm(goal - obs) < self.obstacle_radius + obstacle_margin
                   for obs in self.static_obstacles):
                continue
            break

        return start.astype(np.float32), goal.astype(np.float32)

    def reset(self, randomize=False):
        """
        randomize=False(기본값): __init__에서 정한 고정 시작/목표 좌표 사용 (기존 학습과 동일).
        randomize=True: 매 에피소드 월드 전체 범위에서 시작/목표를 새로 무작위 샘플링
        (시나리오 일반화 테스트용 - 학습 때 못 본 배치에서도 정책이 잘 작동하는지 확인).
        """
        if randomize:
            starts = []
            goals = []
            for _ in range(self.n_agents):
                s, g = self._sample_random_start_goal(starts)
                starts.append(s)
                goals.append(g)
            self.starts = starts
            self.goals = goals

        self.positions = [s.astype(np.float32).copy() for s in self.starts]
        self.headings = [self._wrap_angle(self._bearing_angle(self.goals[i] - self.positions[i])
                                          + self.rng.uniform(-0.3, 0.3))
                          for i in range(self.n_agents)]

        self.done_flags = [False] * self.n_agents
        self.step_count = 0
        self._prev_dist_to_goal = [float(np.linalg.norm(self.goals[i] - self.positions[i]))
                                    for i in range(self.n_agents)]

        self.colreg_checks = 0
        self.colreg_compliant = 0

        self.cr_history = []
        self.colreg_reward_history = []

        return self._get_all_states()

    def step(self, actions):
        """
        actions: 길이 n_agents짜리 리스트, 각 원소는 [steer, speed] (둘 다 [-1,1] 범위)
        반환: states, rewards, terminateds, truncateds, infos (전부 길이 n_agents짜리 리스트)
        """
        self.step_count += 1

        raw_steers = [0.0] * self.n_agents  # COLREG 보상 계산에 쓸, 정규화 전 조향값([-1,1])
        speeds = [0.0] * self.n_agents       # CR(DCPA/TCPA) 계산에 쓸 이번 스텝의 실제 물리 속력

        for i in range(self.n_agents):
            if self.done_flags[i]:
                continue
            raw_steer = float(np.clip(actions[i][0], -1.0, 1.0))
            raw_steers[i] = raw_steer
            steer = raw_steer * self.max_turn_rate
            speed_ratio = (float(np.clip(actions[i][1], -1.0, 1.0)) + 1.0) / 2.0
            speed = speed_ratio * self.max_speed
            speeds[i] = speed

            self.headings[i] = self._wrap_angle(self.headings[i] + steer * self.dt)
            self.positions[i] = self.positions[i] + np.array(
                [np.cos(self.headings[i]), np.sin(self.headings[i])], dtype=np.float32) * speed * self.dt
            self.positions[i] = np.clip(self.positions[i], 0.0, self.world_size)

        rewards = [0.0] * self.n_agents
        terminateds = [False] * self.n_agents
        results = ["running"] * self.n_agents

        for i in range(self.n_agents):
            if self.done_flags[i]:
                results[i] = "already_done"
                continue

            dist_to_goal = float(np.linalg.norm(self.goals[i] - self.positions[i]))
            reward = self.step_penalty
            reward += (self._prev_dist_to_goal[i] - dist_to_goal) * self.progress_scale
            self._prev_dist_to_goal[i] = dist_to_goal

            # ===== 헤딩 보상 (dense): 목표 방향을 바라볼수록 페널티가 작아짐 =====
            _, bearing_goal = self._bearing_from(i, self.goals[i])
            reward -= (abs(bearing_goal) / np.pi) * self.w_heading

            terminated = False
            result = "running"

            for obs_pos in self.static_obstacles:
                if np.linalg.norm(obs_pos - self.positions[i]) < self.obstacle_radius:
                    reward += self.collision_penalty
                    terminated = True
                    result = "collision_obstacle"
                    break

            vel_own = speeds[i] * np.array(
                [np.cos(self.headings[i]), np.sin(self.headings[i])], dtype=np.float64)

            if not terminated:
                # ===== Near-Miss 페널티: DCPA/TCPA 기반 충돌위험도(CR)로 교체 =====
                # 단순 현재거리 대신, "지금 이 속도로 계속 가면 얼마나 위험해지는가"를 반영.
                # 다중 선박이라 합산이 아니라 max(가장 위험한 상대 하나)를 씀 - 스케일 폭발 방지.
                max_cr_this_step = 0.0
                for j in range(self.n_agents):
                    if j == i:
                        continue
                    d_ij = float(np.linalg.norm(self.positions[j] - self.positions[i]))
                    if d_ij < (2 * self.agent_radius):
                        reward += self.collision_penalty
                        terminated = True
                        result = "collision_agent"
                        break
                    vel_target = self.max_speed * np.array(
                        [np.cos(self.headings[j]), np.sin(self.headings[j])], dtype=np.float64)
                    dcpa, tcpa = self._compute_dcpa_tcpa(self.positions[i], vel_own, self.positions[j], vel_target)
                    cr = self._compute_cr(dcpa, tcpa, d_ij)
                    max_cr_this_step = max(max_cr_this_step, cr)

                if not terminated:
                    reward -= self.w_CR * max_cr_this_step
                    self.cr_history.append(max_cr_this_step)

            if not terminated and dist_to_goal < self.goal_radius:
                reward += self.goal_reward
                terminated = True
                result = "goal"

            # ===== COLREG 보상 유도 (CR과 연동: 위험할수록 규칙 준수/위반 신호가 커짐) =====
            if self.colreg_enabled and not terminated:
                colreg_reward_this_agent = 0.0
                for j in range(self.n_agents):
                    if j == i or self.done_flags[j]:
                        continue
                    role = self._classify_colreg(i, j)
                    if role is None:
                        continue

                    self.colreg_checks += 1
                    delta = self._colreg_delta(role, raw_steers[i])
                    if delta > 0:
                        self.colreg_compliant += 1

                    d_ij = float(np.linalg.norm(self.positions[j] - self.positions[i]))
                    vel_target = self.max_speed * np.array(
                        [np.cos(self.headings[j]), np.sin(self.headings[j])], dtype=np.float64)
                    dcpa, tcpa = self._compute_dcpa_tcpa(self.positions[i], vel_own, self.positions[j], vel_target)
                    cr_ij = self._compute_cr(dcpa, tcpa, d_ij)

                    colreg_reward_this_agent += self.w_col * cr_ij * delta

                reward += colreg_reward_this_agent
                self.colreg_reward_history.append(colreg_reward_this_agent)

            rewards[i] = reward
            terminateds[i] = terminated
            results[i] = result
            if terminated:
                self.done_flags[i] = True

        truncated_global = self.step_count >= self.max_steps
        truncateds = [False] * self.n_agents
        for i in range(self.n_agents):
            if truncated_global and not self.done_flags[i]:
                truncateds[i] = True
                results[i] = "timeout"
                self.done_flags[i] = True

        states = self._get_all_states()
        infos = [{"result": results[i], "pos": self.positions[i].copy(), "heading": self.headings[i]}
                 for i in range(self.n_agents)]

        return states, rewards, terminateds, truncateds, infos

    @property
    def all_done(self):
        return all(self.done_flags)

    @property
    def colreg_compliance_rate(self):
        if self.colreg_checks == 0:
            return None
        return self.colreg_compliant / self.colreg_checks

    @property
    def avg_cr(self):
        """이번 에피소드(reset() 이후) 스텝별 max-CR(충돌위험도)의 평균. 로깅용."""
        if not self.cr_history:
            return None
        return float(np.mean(self.cr_history))

    @property
    def avg_colreg_reward(self):
        """이번 에피소드 스텝별 COLREG 보상의 평균. 로깅용."""
        if not self.colreg_reward_history:
            return None
        return float(np.mean(self.colreg_reward_history))

    # ---------- 실시간 시각화 ----------
    def render(self, pause=0.05):
        import matplotlib.pyplot as plt
        if not hasattr(self, "_fig") or self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        ax = self._ax
        ax.clear()

        for obs_pos in self.static_obstacles:
            ax.add_patch(plt.Circle(obs_pos, self.obstacle_radius, color="steelblue", alpha=0.6))

        colors = ["red", "purple", "darkorange", "brown", "black"]
        for i in range(self.n_agents):
            c = colors[i % len(colors)]
            ax.add_patch(plt.Circle(self.goals[i], self.goal_radius, color=c, alpha=0.15))
            marker = "o" if not self.done_flags[i] else "x"
            ax.plot(self.positions[i][0], self.positions[i][1], marker=marker, color=c, markersize=10)
            arrow_len = 0.4
            ax.arrow(self.positions[i][0], self.positions[i][1],
                     arrow_len * np.cos(self.headings[i]), arrow_len * np.sin(self.headings[i]),
                     head_width=0.15, color=c, length_includes_head=True)

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
