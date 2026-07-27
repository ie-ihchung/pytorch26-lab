# ==========================================================
# 미니 프로젝트 1 [기본] — FrozenLake — 미끄러운 얼음호수 건너기
# 환경: FrozenLake-v1 (is_slippery=True)
# 목표: 확률적 전이 환경에서 Q-Learning으로 안전한 경로 정책을 학습한다
# 재사용: 1일차 7교시 CliffWalking 코드에서 환경만 교체
# 사이트: https://pytorch26.dreamitbiz.com/#/projects
# ==========================================================
# [진행 가이드]
#  - is_slippery=False로 먼저 학습해 결정적 환경에서 동작 확인
#  - is_slippery=True로 바꾸면 성공률이 급락하는 것을 관찰
#  - ε 감쇠 스케줄과 학습률 α를 조정해 성공률을 끌어올리기
#  - 학습된 Q 테이블을 화살표 격자로 시각화해 정책 해석
# [결과 인증]
#  평가 모드(ε=0) 1000 에피소드 평균 성공률 70% 이상 + 정책 화살표 격자 공유

import gymnasium as gym
import numpy as np

# 미끄러운 얼음호수: 행동대로 안 움직일 확률이 2/3인 확률적 환경
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
nS, nA = env.observation_space.n, env.action_space.n

alpha, gamma = 0.1, 0.99
eps, eps_min, eps_decay = 1.0, 0.05, 0.9997   # 천천히 감쇠 (확률 환경이라 탐험을 오래)
rng = np.random.default_rng(0)
Q = np.zeros((nS, nA))

# ── 학습 ──
for ep in range(30000):
    s, _ = env.reset()
    done = False
    while not done:
        a = int(rng.integers(nA)) if rng.random() < eps else int(Q[s].argmax())
        s2, r, term, trunc, _ = env.step(a)
        done = term or trunc
        Q[s][a] += alpha * (r + gamma * Q[s2].max() * (not done) - Q[s][a])
        s = s2
    eps = max(eps_min, eps * eps_decay)

# ── 평가 (탐험 없이 greedy) ──
wins = 0
for _ in range(1000):
    s, _ = env.reset()
    done = False
    while not done:
        s, r, term, trunc, _ = env.step(int(Q[s].argmax()))
        done = term or trunc
    wins += (r > 0)
print(f"greedy 성공률: {wins / 10:.1f}%   (70% 이상이면 인증!)")

# ── 정책 화살표 격자 ──
arrows = np.array(['←', '↓', '→', '↑'])   # FrozenLake 행동 순서
print(arrows[Q.argmax(axis=1)].reshape(4, 4))
# 미끄러운 환경에서는 "구멍 반대쪽으로 미는" 이상해 보이는
# 화살표가 오히려 최적일 수 있습니다 — 왜인지 생각해 보세요
