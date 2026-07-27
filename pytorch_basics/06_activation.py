# ============================================================
# [파이토치 기초 ⑥] 활성화 함수 — 직선을 구부리는 장치
# ------------------------------------------------------------
# 왜 필요한가:
#   nn.Linear 는 직선입니다. 직선을 아무리 여러 번 겹쳐도 직선입니다.
#     y = w2(w1·x + b1) + b2 = (w2·w1)x + (...)  ← 결국 직선 하나
#   그래서 층 사이에 '구부리는 장치'를 하나씩 끼웁니다. 그게 활성화 함수입니다.
#
# 실행: python 06_activation.py    (2초)
# ============================================================
import torch
import torch.nn as nn

print('=' * 55)
print('1. 직선만 쌓으면 정말 직선인지 확인')
print('=' * 55)

torch.manual_seed(0)
no_act = nn.Sequential(nn.Linear(1, 8), nn.Linear(8, 8), nn.Linear(8, 1))

x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
with torch.no_grad():
    y = no_act(x).squeeze()

print('  x       :', [f'{v:6.2f}' for v in x.squeeze().tolist()])
print('  출력    :', [f'{v:6.3f}' for v in y.tolist()])
diffs = [round((y[i + 1] - y[i]).item(), 4) for i in range(len(y) - 1)]
print('  이웃 차이:', diffs)
print('  → 차이가 전부 같습니다. 층을 3개 쌓았는데도 그냥 직선입니다.')

print()
print('=' * 55)
print('2. 대표 활성화 함수 4종')
print('=' * 55)

vals = torch.tensor([-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0])
print(f'  {"입력":>8}', ''.join(f'{v:8.1f}' for v in vals.tolist()))
print('  ' + '-' * 64)
for name, fn in [('ReLU', torch.relu),
                 ('Sigmoid', torch.sigmoid),
                 ('Tanh', torch.tanh),
                 ('Softplus', nn.functional.softplus)]:
    out = fn(vals)
    print(f'  {name:>8}', ''.join(f'{v:8.3f}' for v in out.tolist()))

print('''
  ReLU     : 음수는 0, 양수는 그대로. 가장 많이 씁니다. 빠르고 잘 됩니다.
  Sigmoid  : 0~1 사이로 눌러 줍니다. "확률"이 필요할 때.
  Tanh     : -1~1 사이. 3일차 DDPG/SAC 에서 행동 범위를 맞출 때 씁니다.
  Softplus : ReLU 를 부드럽게 만든 것. 항상 양수 → 표준편차 만들 때.''')

print()
print('=' * 55)
print('3. 왜 요즘은 ReLU 를 기본으로 쓰나 — 기울기 소실')
print('=' * 55)

# 층을 깊게 쌓았을 때 기울기가 살아 남는지 비교합니다.
for name, act in [('Sigmoid', nn.Sigmoid), ('Tanh', nn.Tanh), ('ReLU', nn.ReLU)]:
    torch.manual_seed(0)
    layers = []
    for _ in range(10):                       # 10층을 쌓는다
        layers += [nn.Linear(16, 16), act()]
    net = nn.Sequential(*layers, nn.Linear(16, 1))

    inp = torch.randn(8, 16)
    out = net(inp).sum()
    out.backward()

    first = net[0].weight.grad.abs().mean().item()   # 맨 앞 층에 도달한 기울기 크기
    print(f'  {name:8s} 10층 통과 후 맨 앞 층의 기울기 크기 = {first:.3e}')

print('''
  → Sigmoid 는 기울기가 거의 0이 됩니다 (기울기 소실).
    앞쪽 층이 배우지를 못합니다.
  → ReLU 는 기울기를 그대로 통과시켜 깊게 쌓아도 살아 있습니다.''')

print()
print('=' * 55)
print('4. 강화학습에서 마지막 층에 활성화를 쓰지 않는 이유')
print('=' * 55)

print('''  Q값은 -100 일 수도 +500 일 수도 있습니다.
  마지막에 Sigmoid 를 붙이면 0~1 로 눌려 버려 절대 표현할 수 없습니다.

    (X) nn.Linear(64, 2), nn.Sigmoid()
    (O) nn.Linear(64, 2)                ← 그냥 끝낸다

  정책망(Policy)도 마찬가지입니다. 마지막은 그냥 '점수(logit)'로 두고,
  확률이 필요하면 Categorical(logits=...) 이 안에서 softmax 를 합니다.

  예외: 3일차 DDPG/SAC 는 행동이 [-2, 2] 처럼 범위가 정해져 있어서
        마지막에 Tanh 를 붙이고 범위만큼 곱해 줍니다.''')

# ── 그림 ──────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    xs = torch.linspace(-4, 4, 200)
    plt.figure(figsize=(7, 4.5))
    for name, fn in [('ReLU', torch.relu), ('Sigmoid', torch.sigmoid),
                     ('Tanh', torch.tanh), ('Softplus', nn.functional.softplus)]:
        plt.plot(xs.numpy(), fn(xs).numpy(), lw=2, label=name)
    plt.axhline(0, color='k', lw=.8); plt.axvline(0, color='k', lw=.8)
    plt.ylim(-1.5, 3); plt.grid(alpha=.3); plt.legend()
    plt.title('Activation functions'); plt.tight_layout(); plt.show()
except ImportError:
    pass

# ============================================================
# 바꿔 보기
#   1) 2번 실험의 층 수를 20으로 늘리면 Sigmoid 는 어떻게 되나요?
#   2) ReLU 대신 nn.LeakyReLU() 를 써 보세요 (음수도 조금 통과시킵니다)
#   3) 1번 실험에 nn.ReLU() 를 끼워 넣고 다시 돌려 보세요.
#      이웃 차이가 더 이상 일정하지 않게 됩니다 = 구부러졌다는 뜻입니다.
# ============================================================
