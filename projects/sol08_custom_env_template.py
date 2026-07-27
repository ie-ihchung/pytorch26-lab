# ==========================================================
# 미니 프로젝트 8 [자유] — 토이 프로젝트 — 나만의 강화학습 문제 풀기
# 환경: 자유 (커스텀 Gym 환경 / Unity ML-Agents / Atari 등)
# 목표: 3일간 배운 알고리즘 중 하나를 골라 스스로 정의한 문제에 적용한다
# 재사용: 과정 전체 코드 중 자유 선택
# 사이트: https://pytorch26.dreamitbiz.com/#/projects
# ==========================================================
# [진행 가이드]
#  - 문제 정의: 상태·행동·보상을 직접 설계 (gymnasium.Env 상속 커스텀 환경 추천)
#  - 알고리즘 선택 근거 정리 (이산/연속 행동, 보상 밀도 기준)
#  - 학습 → 실패 원인 분석 → 보상·하이퍼파라미터 수정의 반복 과정을 기록
#  - Unity ML-Agents로 3D 환경을 만들면 시각적으로 인상적인 결과물이 됩니다
# [결과 인증]
#  문제 정의 + 학습 결과(곡선/영상) + 시행착오 기록 공유

# 커스텀 Gym 환경 템플릿 — "술래잡기 격자" 예시
# 에이전트(A)가 도망다니는 목표(T)를 잡는 5x5 그리드
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridChaseEnv(gym.Env):
    """상태: (내 r,c, 목표 r,c) / 행동: 상하좌우 / 보상: 잡으면 +10, 스텝 -0.1"""
    def __init__(self, size=5, max_steps=50):
        super().__init__()
        self.size, self.max_steps = size, max_steps
        self.observation_space = spaces.Box(0, size - 1, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent = self.np_random.integers(0, self.size, 2)
        self.target = self.np_random.integers(0, self.size, 2)
        self.steps = 0
        return self._obs(), {}

    def _obs(self):
        return np.concatenate([self.agent, self.target]).astype(np.float32)

    def step(self, action):
        move = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        self.agent = np.clip(self.agent + move, 0, self.size - 1)
        # 목표는 30% 확률로 무작위 도망
        if self.np_random.random() < 0.3:
            t = [(-1, 0), (1, 0), (0, -1), (0, 1)][self.np_random.integers(4)]
            self.target = np.clip(self.target + t, 0, self.size - 1)
        self.steps += 1
        caught = bool(np.array_equal(self.agent, self.target))
        reward = 10.0 if caught else -0.1
        return self._obs(), reward, caught, self.steps >= self.max_steps, {}

# ── 만든 환경을 표 기반 Q-Learning으로 빠르게 검증 ──
env = GridChaseEnv()
n_act = env.action_space.n

def key(obs):   # 격자 좌표라 정수 튜플을 테이블 키로 사용
    return tuple(obs.astype(int))

Q = {}
rng = np.random.default_rng(0)
eps = 1.0
for ep in range(20000):
    s, _ = env.reset(seed=ep)
    done = False
    while not done:
        k = key(s)
        Q.setdefault(k, np.zeros(n_act))
        a = int(rng.integers(n_act)) if rng.random() < eps else int(Q[k].argmax())
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        k2 = key(s2)
        Q.setdefault(k2, np.zeros(n_act))
        Q[k][a] += 0.1 * (r + 0.99 * Q[k2].max() * (not done) - Q[k][a])
        s = s2
    eps = max(0.05, eps * 0.9997)

# ── 평가 ──
wins = 0
for i in range(100):
    s, _ = env.reset(seed=90000 + i)
    done = False
    while not done:
        a = int(Q.get(key(s), np.zeros(n_act)).argmax())
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
    wins += term
print(f"잡기 성공률: {wins}%")
# 다음 단계: 보상·규칙을 바꿔 보고, DQN(2일차 코드)으로 교체해 보세요
