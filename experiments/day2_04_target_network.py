# ============================================================
# [2일차 4교시 실습] 타깃 네트워크가 있으면 뭐가 달라지나 — 학습곡선 비교
# ------------------------------------------------------------
# 같은 DQN 코드를 두 번 돌립니다. 딱 한 곳만 다릅니다.
#   ① 타깃 네트워크 있음 : 정답(과녁)을 잠시 고정해 두고 맞춘다
#   ② 타깃 네트워크 없음 : 지금 학습 중인 그 신경망으로 정답도 만든다
#
# ②는 "과녁을 든 사람이 같이 뛰는" 상황입니다. 맞추려고 다가가면
# 과녁도 같이 움직여서 학습이 출렁입니다.
#
# 실행 시간: 2~4분 (CPU) — 학습곡선 2개가 나옵니다
# ============================================================
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym

SEED = 0
EPISODES = 250          # 에피소드 수 (늘리면 더 뚜렷해지지만 오래 걸립니다)
GAMMA = 0.99            # 미래를 얼마나 챙길지
BATCH = 64              # 한 번에 몇 개의 경험으로 배울지
LR = 1e-3               # 학습률
TARGET_SYNC = 200       # 몇 스텝마다 과녁을 갱신할지


class QNet(nn.Module):
    """상태 4개를 받아 행동 2개의 Q값을 내놓는 아주 작은 신경망."""
    def __init__(self, n_obs, n_act):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(n_obs, 128), nn.ReLU(),
            nn.Linear(128, n_act),
        )

    def forward(self, x):
        return self.f(x)


def run(use_target):
    """use_target=True 면 타깃 네트워크를 쓰고, False 면 안 쓴다."""
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    env = gym.make('CartPole-v1')
    n_obs = env.observation_space.shape[0]      # 4
    n_act = env.action_space.n                  # 2

    q = QNet(n_obs, n_act)                      # 학습하는 본체
    tgt = QNet(n_obs, n_act)                    # 과녁 (본체의 복사본)
    tgt.load_state_dict(q.state_dict())         # 출발은 똑같이

    opt = torch.optim.Adam(q.parameters(), lr=LR)
    buf = deque(maxlen=20000)                   # 경험 저장고
    eps, step = 1.0, 0                          # 탐험 확률, 총 스텝
    scores = []                                 # 에피소드별 점수 (= 학습곡선)

    for ep in range(EPISODES):
        s, _ = env.reset(seed=SEED + ep)
        total, done = 0.0, False

        while not done:
            # ── 행동 고르기 (ε-greedy) ──────────────────
            if random.random() < eps:
                a = env.action_space.sample()               # 가끔 아무거나 (탐험)
            else:
                with torch.no_grad():                       # 행동 고를 땐 기울기 불필요
                    a = int(q(torch.tensor(s)).argmax())    # 보통은 Q가 가장 큰 행동

            s2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            buf.append((s, a, r, s2, float(term)))          # 경험 한 줄 저장
            s, total, step = s2, total + r, step + 1

            eps = max(0.05, eps * 0.9995)                   # 탐험은 점점 줄인다

            # ── 배우기 ─────────────────────────────────
            if len(buf) >= 1000:                            # 조금 모인 뒤부터
                batch = random.sample(buf, BATCH)           # 섞어서 꺼낸다 (경험 재현)
                bs, ba, br, bs2, bd = map(np.array, zip(*batch))
                bs = torch.tensor(bs, dtype=torch.float32)
                bs2 = torch.tensor(bs2, dtype=torch.float32)
                ba = torch.tensor(ba, dtype=torch.int64).unsqueeze(1)
                br = torch.tensor(br, dtype=torch.float32).unsqueeze(1)
                bd = torch.tensor(bd, dtype=torch.float32).unsqueeze(1)

                pred = q(bs).gather(1, ba)                  # 내가 실제로 한 행동의 Q

                with torch.no_grad():                       # ★ 여기가 이 실습의 전부 ★
                    if use_target:
                        nxt = tgt(bs2).max(1, keepdim=True)[0]   # 과녁은 고정된 네트워크
                    else:
                        nxt = q(bs2).max(1, keepdim=True)[0]     # 과녁도 지금 이 네트워크
                    target = br + GAMMA * nxt * (1 - bd)         # 정답 = 지금 점수 + 다음 값

                loss = nn.functional.smooth_l1_loss(pred, target)
                opt.zero_grad(); loss.backward(); opt.step()

                if use_target and step % TARGET_SYNC == 0:
                    tgt.load_state_dict(q.state_dict())     # 가끔 과녁을 갱신

        scores.append(total)
        if (ep + 1) % 50 == 0:
            print(f'    에피소드 {ep+1:3d}  최근 50판 평균 {np.mean(scores[-50:]):6.1f}')

    env.close()
    return scores


def smooth(x, k=20):
    """들쭉날쭉한 점수를 k판 이동평균으로 부드럽게 (흐름을 보기 위함)"""
    return [np.mean(x[max(0, i - k):i + 1]) for i in range(len(x))]


print('① 타깃 네트워크 있음 — 학습 중...')
with_t = run(use_target=True)
print('\n② 타깃 네트워크 없음 — 학습 중...')
without_t = run(use_target=False)

print('\n  === 결과 ===')
print(f'  타깃 있음  마지막 50판 평균: {np.mean(with_t[-50:]):6.1f}   전체 최고: {max(with_t):.0f}')
print(f'  타깃 없음  마지막 50판 평균: {np.mean(without_t[-50:]):6.1f}   전체 최고: {max(without_t):.0f}')
print('\n  → 타깃이 없으면 학습이 아예 안 되거나, 올라가다 무너집니다.')
print('  → 강사 맥북 실측: 있음 115.6점 / 없음 10.7점 (없는 쪽은 처음보다 오히려 나빠졌습니다).')
print('  → 숫자는 실행할 때마다 다릅니다. 방향만 같으면 정상입니다.')

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 4.5))
    plt.plot(with_t, alpha=.25, color='tab:blue')
    plt.plot(without_t, alpha=.25, color='tab:red')
    plt.plot(smooth(with_t), color='tab:blue', lw=2, label='with target network')
    plt.plot(smooth(without_t), color='tab:red', lw=2, label='without target network')
    plt.xlabel('episode'); plt.ylabel('return')
    plt.title('Does the target network matter?')
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.show()
except ImportError:
    print('\n  (matplotlib 이 없어 그림은 건너뜁니다)')

# ============================================================
# 바꿔 보기
#   1) TARGET_SYNC = 20 으로 줄이면? → 과녁이 자주 움직여 '없음'에 가까워집니다
#   2) TARGET_SYNC = 2000 으로 늘리면? → 안정적이지만 학습이 느려집니다
#   3) EPISODES = 500 으로 늘리면 차이가 더 확실해집니다 (시간은 2배)
#   4) SEED 를 1, 2, 3 으로 바꿔 여러 번 돌려 보세요.
#      강화학습은 한 번의 결과로 판단하면 안 됩니다.
# ============================================================
