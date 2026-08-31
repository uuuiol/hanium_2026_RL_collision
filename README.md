# 멀티에이전트 PPO + COLREG 자율운항 시뮬레이터

## 이 폴더가 뭔가요

강화학습(PPO)으로 **배 3척이 서로를 피하면서 각자의 목적지까지 항해**하는 걸 학습시키는 코드입니다.
국제해상충돌예방규칙(COLREG)까지 반영되어 있어서, 배들이 규칙에 맞게 우현으로 피하는 등의
행동도 학습합니다.

---

## 1. 설치 (한 번만 하면 됨)

```
pip install tensorflow numpy matplotlib
```
- Python 3.10 이상 권장
- GPU 없어도 돌아갑니다 (CPU로 충분)

---

## 2. 파일 목록 및 역할

### 핵심 파일 (이 4개가 메인)

| 파일 | 역할 | 직접 실행? |
|---|---|---|
| `ppo_model.py` | PPO 신경망(두뇌) 구조 정의 | ❌ |
| `multi_boat_env.py` | 배 3척이 사는 세상 (장애물, COLREG, 안전거리 판정) | ❌ |
| `multi_ppo_train_logging.py` | **학습 실행** (위 두 개를 조합해서 실제로 훈련) | ✅ 이걸 실행 |
| `watch_multi_trained_agents.py` | **결과 구경** (학습된 배들이 움직이는 걸 실시간으로 봄) | ✅
 학습 끝난 뒤 실행 |

### 보조 파일

| 파일 | 역할 |
|---|---|
| `ppo_continuous_train_logging.py` | PPOAgent 클래스 정의 (multi_ppo가 이걸 가져다 씀) |
| `boat_env.py` | 배 1척용 환경 (ppo_continuous가 의존) |
| `watch_trained_agent.py` | 배 1척용 구경 스크립트 (멀티용과 혼동 주의!) |
| `top10_episodes.py` | 학습 결과 CSV에서 점수 상위 10개 뽑아 보는 도구 |

---

## 3. 실행 순서

### STEP 1: 학습시키기
```
python multi_ppo_train_logging.py
```
- 400 에피소드 학습 (로컬 PC 기준 20~40분 소요)
- 진행상황이 10 에피소드마다 콘솔에 출력됨
- ★ 표시와 함께 "신기록!" 이 뜨면, 그 시점의 가중치가 자동 저장되는 것
- **끝나면 생기는 파일들:**
  - `multi_ppo_results.csv` — 에피소드별 결과 표
  - `multi_ppo_training_result.png` — 학습 곡선 그래프
  - `multi_ppo_model_weights.weights.h5` — 마지막 시점 가중치
  - `multi_ppo_model_weights_best.weights.h5` — **최고 성능 시점 가중치 (이걸 쓰세요!)**

### STEP 2: 결과 구경하기
`watch_multi_trained_agents.py`를 열어서 아래 한 줄을 확인:
```python
WEIGHTS_PATH = "multi_ppo_model_weights.weights.h5"
```
이걸 **best 버전으로 바꿔주세요:**
```python
WEIGHTS_PATH = "multi_ppo_model_weights_best.weights.h5"
```
그 다음 실행:
```
python watch_multi_trained_agents.py
```
- 30판을 빠르게 스캔한 뒤, **목표 도달한 배가 가장 많은 판을 골라서** 상위 5개만 실시간으로 보여줌
- 각 판 재생 전에 배별 결과(✅ 성공 / 💥 충돌 / ⏱ 시간초과)가 콘솔에 표시됨

### STEP 3: (선택) 점수 상위 10개 확인
```
python top10_episodes.py
```
- `multi_ppo_results.csv`에서 점수 높은 에피소드 10개를 표로 보여줌

---

## 4. 파라미터 바꿔서 실험하고 싶을 때

### 학습 파라미터 (`multi_ppo_train_logging.py` 상단)
| 변수 | 기본값 | 설명 |
|---|---|---|
| `NUM_WORLD_EPISODES` | 400 | 총 몇 판 학습시킬지 (늘리면 더 잘 배우지만 오래 걸림) |
| `N_AGENTS` | 3 | 배 몇 척 (2로 줄이면 더 빨리 수렴) |
| `MAX_STEPS` | 200 | 한 판 최대 스텝 수 |
| `RESULT_PREFIX` | "multi_ppo" | 결과 파일 이름 (⚠️ 바꿔야 이전 결과 안 덮어씀!) |

### 환경 파라미터 (`multi_boat_env.py` 생성자)
| 변수 | 기본값 | 설명 |
|---|---|---|
| `colreg_enabled` | True | False로 끄면 COLREG 없이 학습 (비교 실험용) |
| `safety_distance` | 0.7 | CPA 안전거리 (줄이면 더 바짝 붙어도 됨) |
| `goal_radius` | 0.3 | 목표 반경 (0.4~0.5로 늘리면 도달 판정이 관대해짐) |
| `collision_penalty` | -10.0 | 충돌 시 벌점 크기 |

---

## 5. 주의사항

- **`watch_trained_agent.py`(1척용)와 `watch_multi_trained_agents.py`(3척용)를 절대 혼동하지 마세요!**
  멀티로 학습한 가중치를 1척용 watch로 불러오면 신경망 크기가 안 맞아서 에러가 납니다.

- 파라미터를 바꿔서 여러 번 돌릴 때는 **반드시 `RESULT_PREFIX`를 매번 다르게** 해주세요.
  안 그러면 이전 결과 파일(CSV, 그래프, 가중치)이 덮어써집니다.
  예: `RESULT_PREFIX = "multi_ppo_lr0001"`, `RESULT_PREFIX = "multi_ppo_colreg_off"` 등

- 학습 도중 강제 종료(Ctrl+C)해도 **CSV는 이미 저장되어 있습니다** (매 에피소드마다 즉시 기록).
  다만 가중치 파일은 마지막 저장 시점까지만 남아있습니다.

---

## 6. 핵심 개선사항 (기존 대비 바뀐 점)

1. **안전거리(CPA) 페널티**: 충돌 안 해도 다른 배에 너무 가까이 가면 거리 비례 페널티
   → COLREG 제8조 "충분히 여유 있는 거리를 두고 피항" 반영

2. **최고 성능 시점 자동 저장(Best Checkpoint)**: 학습 중 성능이 최고치를 찍을 때마다
   별도 파일로 저장 → 후반에 성능이 떨어져도 "제일 좋았던 순간"의 정책을 보존

3. **학습률 감쇠(LR Decay)**: 학습 초반엔 과감하게(3e-4), 후반엔 미세조정(5e-5)
   → 후반부 성능 붕괴(catastrophic forgetting) 완화

4. **장애물 공정 배치**: 세 배의 경로가 장애물에 대해 비슷한 여유를 갖도록 좌표 재계산
   → 특정 배만 구조적으로 불리했던 문제 해결
