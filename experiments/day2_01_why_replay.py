# ============================================================
# [2일차 1교시 실습] 경험 재현(Replay)이 왜 필요한가 — 눈으로 보기
# ------------------------------------------------------------
# 강화학습 코드를 쓰지 않습니다. 아주 단순한 곡선 맞추기 문제로
# "순서대로 준 데이터"와 "섞어서 준 데이터"가 어떻게 다른지만 봅니다.
#
# 왜 이 실험인가:
#   강화학습에서 에이전트가 겪는 경험은 앞뒤가 딱 붙어 있습니다.
#   (왼쪽으로 갔다 → 또 왼쪽 → 또 왼쪽 ...)
#   이렇게 비슷한 것만 연달아 주면 신경망은 "방금 본 것"에만 맞추고
#   예전에 배운 것을 잊습니다. 그래서 버퍼에 쌓아 두고 섞어서 꺼냅니다.
#
# 실행 시간: 20초 안팎 (CPU)
# ============================================================
import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(0)          # 같은 결과가 나오도록 고정
np.random.seed(0)

# ── 문제 만들기 ───────────────────────────────────────────
# x 는 -3 부터 3 까지 600개. 정답은 sin(x).
# 신경망이 이 곡선을 외우게 하는 것이 목표입니다.
X = np.linspace(-3, 3, 600, dtype=np.float32)
Y = np.sin(X)

X_t = torch.tensor(X).unsqueeze(1)      # (600, 1) — 세로로 세운다
Y_t = torch.tensor(Y).unsqueeze(1)      # (600, 1)


def make_net():
    """작은 신경망 하나. 1개 들어와서 1개 나간다."""
    torch.manual_seed(0)                # 두 실험의 출발점을 똑같이
    return nn.Sequential(
        nn.Linear(1, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 1),
    )


def train(shuffle, steps=1500, batch=32):
    """
    shuffle=False : 데이터를 앞에서부터 순서대로 32개씩 준다 (강화학습의 날것 경험)
    shuffle=True  : 600개 중에서 무작위로 32개를 뽑아 준다 (경험 재현)
    반환값: 학습 도중 '전체 데이터'에 대한 오차 기록
    """
    net = make_net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    curve = []                                  # 학습곡선용 기록
    cursor = 0                                  # 순서대로 줄 때 현재 위치

    for step in range(steps):
        if shuffle:
            idx = torch.randint(0, len(X_t), (batch,))     # 무작위로 32개
        else:
            idx = torch.arange(cursor, cursor + batch) % len(X_t)  # 붙어 있는 32개
            cursor = (cursor + batch) % len(X_t)           # 다음 번엔 그 다음 구간

        pred = net(X_t[idx])                    # ① 예측
        loss = nn.functional.mse_loss(pred, Y_t[idx])      # ② 이번 조각의 오차
        opt.zero_grad()                         # ③ 지난 기울기 지우기
        loss.backward()                         # ④ 기울기 계산
        opt.step()                              # ⑤ 살짝 이동

        if step % 50 == 0:
            with torch.no_grad():               # 채점만 할 땐 기울기 필요 없음
                whole = nn.functional.mse_loss(net(X_t), Y_t).item()
            curve.append((step, whole))         # 전체를 얼마나 맞추는가
    return net, curve


print('학습 중... 20초쯤 걸립니다\n')
net_seq, curve_seq = train(shuffle=False)   # 순서대로
net_shf, curve_shf = train(shuffle=True)    # 섞어서

# ── 숫자로 보기 ───────────────────────────────────────────
print('     step |  순서대로 준 경우 |  섞어서 준 경우')
print('  --------+------------------+-----------------')
for (s, a), (_, b) in zip(curve_seq, curve_shf):
    if s % 250 == 0:
        print(f'  {s:7d} | {a:16.4f} | {b:15.4f}')

print(f'\n  최종 오차  순서대로: {curve_seq[-1][1]:.4f}   섞어서: {curve_shf[-1][1]:.4f}')
print('  → 섞어서 준 쪽이 훨씬 작습니다. 이것이 경험 재현을 쓰는 이유입니다.')

# ── 그림으로 보기 ─────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # 왼쪽: 학습곡선 (오차가 줄어드는 모양)
    ax[0].plot(*zip(*curve_seq), label='sequential (no replay)')
    ax[0].plot(*zip(*curve_shf), label='shuffled (replay)')
    ax[0].set_yscale('log')                    # 차이가 커서 로그 눈금이 잘 보인다
    ax[0].set_xlabel('step'); ax[0].set_ylabel('MSE on ALL data (log)')
    ax[0].set_title('Learning curve'); ax[0].legend(); ax[0].grid(alpha=.3)

    # 오른쪽: 실제로 그린 곡선 모양
    with torch.no_grad():
        ax[1].plot(X, Y, 'k--', label='target sin(x)')
        ax[1].plot(X, net_seq(X_t).squeeze().numpy(), label='sequential')
        ax[1].plot(X, net_shf(X_t).squeeze().numpy(), label='shuffled')
    ax[1].set_title('What the network learned'); ax[1].legend(); ax[1].grid(alpha=.3)

    plt.tight_layout()
    plt.show()
except ImportError:
    print('\n  (matplotlib 이 없어 그림은 건너뜁니다)')

# ============================================================
# 바꿔 보기
#   1) batch=32 를 8 로 줄이면? → 순서대로 준 쪽이 더 심하게 흔들립니다
#   2) steps 를 5000 으로 늘리면? → 순서대로 준 쪽도 좋아지지만 여전히 느립니다
#   3) train(shuffle=False) 의 cursor 이동을 없애 같은 구간만 반복해서 주면?
#      → 그 구간만 완벽하게 맞추고 나머지는 엉망이 됩니다 (= 망각)
# ============================================================
