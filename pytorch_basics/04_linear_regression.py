# ============================================================
# [파이토치 기초 ④] 선형회귀 — 학습 5단계를 처음부터 끝까지
# ------------------------------------------------------------
# 가장 단순한 학습입니다. 점들이 흩어져 있고, 그 사이를 지나는
# 직선 하나를 찾습니다. y = wx + b 에서 w 와 b 를 찾는 것입니다.
#
# 여기서 나오는 5단계가 앞으로 모든 코드에 그대로 반복됩니다:
#   ① 예측  ② 손실  ③ 기울기 지우기  ④ 역전파  ⑤ 한 걸음
#
# 실행: python 04_linear_regression.py    (5초)
# ============================================================
import torch
import torch.nn as nn

torch.manual_seed(0)            # 결과 고정

print('=' * 55)
print('1. 데이터 만들기 — 정답은 y = 3x + 2 입니다')
print('=' * 55)

x = torch.linspace(-3, 3, 100).unsqueeze(1)      # (100, 1) — 세로로 세운 100개
y = 3 * x + 2 + torch.randn(x.shape) * 0.5       # 정답에 잡음을 조금 섞는다

print('x 모양 =', x.shape, '  y 모양 =', y.shape)
print('  ※ (100, 1) 은 "100개의 데이터, 각각 값 1개" 라는 뜻입니다.')
print('    (100,) 이 아니라 (100, 1) 이어야 합니다 — 신경망은 표를 받습니다.')

print()
print('=' * 55)
print('2. 모델 — 직선 하나')
print('=' * 55)

model = nn.Linear(1, 1)         # 값 1개 받아 값 1개 내놓음 = y = wx + b
print('처음 w =', round(model.weight.item(), 3),
      '  b =', round(model.bias.item(), 3), ' (무작위로 시작)')

print()
print('=' * 55)
print('3. 옵티마이저와 손실 함수 고르기')
print('=' * 55)

optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
#                           ^^^^^^^^^^^^^^^^^^   고칠 대상(w, b)을 알려 준다
#                                                lr = 한 번에 얼마나 움직일지
criterion = nn.MSELoss()        # 틀린 정도를 재는 자
print('옵티마이저: SGD(lr=0.05)   손실: MSE')

print()
print('=' * 55)
print('4. 학습 — 이 5줄이 전부입니다')
print('=' * 55)

for epoch in range(200):
    pred = model(x)                    # ① 예측한다
    loss = criterion(pred, y)          # ② 얼마나 틀렸나 잰다
    optimizer.zero_grad()              # ③ 지난 기울기를 지운다  ← 빠뜨리면 누적된다
    loss.backward()                    # ④ 어느 쪽으로 고칠지 계산한다
    optimizer.step()                   # ⑤ 그 방향으로 한 걸음 간다

    if epoch % 40 == 0:
        print(f'  epoch {epoch:3d}   손실 {loss.item():7.4f}   '
              f'w {model.weight.item():6.3f}   b {model.bias.item():6.3f}')

print(f'\n  최종      w = {model.weight.item():.3f}   b = {model.bias.item():.3f}')
print('  정답      w = 3.000       b = 2.000')
print('  → 거의 맞혔습니다. 잡음을 섞었으니 완전히 같지는 않습니다.')

print()
print('=' * 55)
print('5. 예측해 보기')
print('=' * 55)

with torch.no_grad():                  # 예측만 할 땐 기울기 계산이 필요 없다
    for v in [0.0, 1.0, 5.0]:
        p = model(torch.tensor([[v]])).item()
        print(f'  x = {v:4.1f}  →  예측 {p:6.3f}   (정답 {3*v+2:.3f})')

print('\n  ※ x=5 는 학습에 없던 값입니다(-3~3만 배웠음).')
print('    그래도 잘 맞히는 이유는 직선이라는 구조를 배웠기 때문입니다.')

# ── 그림 ──────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    with torch.no_grad():
        plt.figure(figsize=(6, 4.5))
        plt.scatter(x.numpy(), y.numpy(), s=12, alpha=.5, label='data')
        plt.plot(x.numpy(), model(x).numpy(), 'r-', lw=2, label='learned line')
        plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(alpha=.3)
        plt.title('Linear regression'); plt.tight_layout(); plt.show()
except ImportError:
    pass

# ============================================================
# 바꿔 보기
#   1) lr=0.5 로 키우면? → 너무 크게 움직여 손실이 튀거나 발산합니다
#   2) lr=0.001 로 줄이면? → 200번으로는 다 못 갑니다 (느림)
#   3) optimizer.zero_grad() 를 지우면? → 기울기가 쌓여 엉뚱한 곳으로 갑니다
#      (실제로 지워 보고 무슨 일이 나는지 꼭 한 번 보세요)
#   4) SGD 대신 torch.optim.Adam 으로 바꾸면? → 훨씬 빨리 수렴합니다
# ============================================================
