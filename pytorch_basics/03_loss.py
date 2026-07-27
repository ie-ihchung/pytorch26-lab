# ============================================================
# [파이토치 기초 ③] 손실 함수 — "얼마나 틀렸는가"를 숫자 하나로
# ------------------------------------------------------------
# 학습은 '틀린 정도'를 줄이는 일입니다. 그 틀린 정도를 재는 자가
# 손실 함수(loss function)입니다.
#
# 문제 종류에 따라 쓰는 자가 다릅니다:
#   숫자를 맞히는 문제(회귀)   → MSE
#   분류하는 문제(고양이/개)   → CrossEntropy
#   강화학습의 Q값 맞히기      → MSE 또는 SmoothL1(=Huber)
#
# 실행: python 03_loss.py    (1초)
# ============================================================
import torch
import torch.nn as nn

print('=' * 55)
print('1. MSE — 평균제곱오차 (숫자를 맞히는 문제)')
print('=' * 55)

pred = torch.tensor([3.0, 5.0, 2.0])        # 내가 예측한 값
true = torch.tensor([3.0, 4.0, 5.0])        # 정답

# 손으로 계산해 보기
diff = pred - true                          # 차이:  0,  1, -3
print('차이        =', diff)
print('차이의 제곱 =', diff ** 2)           # 0, 1, 9  (부호를 없애려고 제곱)
print('그 평균     =', (diff ** 2).mean().item(), ' ← 이것이 MSE')

# 파이토치가 해주는 것
print('nn.functional.mse_loss →', nn.functional.mse_loss(pred, true).item())
print('  ※ 제곱을 하므로 크게 틀린 것에 훨씬 큰 벌점이 붙습니다.')

print()
print('=' * 55)
print('2. SmoothL1(Huber) — 강화학습이 즐겨 쓰는 자')
print('=' * 55)

# 강화학습은 가끔 말도 안 되게 튀는 값이 섞입니다.
# MSE 는 제곱이라 그 하나 때문에 학습이 통째로 흔들립니다.
# SmoothL1 은 크게 틀린 구간에서 벌점을 완만하게 줍니다.
for d in [0.5, 1.0, 3.0, 10.0]:
    p = torch.tensor([d]); t = torch.tensor([0.0])
    m = nn.functional.mse_loss(p, t).item()
    s = nn.functional.smooth_l1_loss(p, t).item()
    print(f'  차이 {d:5.1f} →  MSE {m:8.2f}   SmoothL1 {s:7.2f}')
print('  → 차이 10일 때 MSE는 100, SmoothL1은 9.5. 튀는 값에 덜 휘둘립니다.')
print('  ※ 2일차 DQN 코드에서 smooth_l1_loss 를 쓰는 이유가 이것입니다.')

print()
print('=' * 55)
print('3. CrossEntropy — 분류 문제')
print('=' * 55)

# 3개 중 하나를 고르는 문제. 신경망은 '점수(logit)'를 내놓습니다.
logits = torch.tensor([[2.0, 0.5, 0.1]])     # 0번이 가장 그럴듯하다고 본 상태
label = torch.tensor([0])                    # 정답도 0번

print('신경망이 낸 점수(logits) =', logits)
probs = torch.softmax(logits, dim=1)         # 점수 → 확률로 변환
print('확률로 바꾸면            =', probs)
print('  (softmax: 모두 양수로 만들고 합이 1이 되게 나눈다)')

loss = nn.functional.cross_entropy(logits, label)
print('\n정답이 0번일 때 손실 =', round(loss.item(), 4), ' (작다 = 잘 맞혔다)')

label_wrong = torch.tensor([2])              # 정답이 2번이었다면?
loss_w = nn.functional.cross_entropy(logits, label_wrong)
print('정답이 2번일 때 손실 =', round(loss_w.item(), 4), ' (크다 = 많이 틀렸다)')

print('\n  ※ cross_entropy 는 softmax 를 안에서 알아서 합니다.')
print('    직접 softmax 를 씌운 뒤 또 넣으면 두 번 적용되어 학습이 안 됩니다.')

print()
print('=' * 55)
print('4. 정책 경사(Policy Gradient)의 손실은 왜 음수가 붙나')
print('=' * 55)

# 5교시에 나오는 loss = -(log_prob * G) 를 여기서 미리 봅니다.
logits = torch.tensor([[1.0, 1.0]], requires_grad=True)   # 두 행동이 반반
dist = torch.distributions.Categorical(logits=logits)
a = torch.tensor([0])                                     # 0번 행동을 했다고 치자
logp = dist.log_prob(a)

print('두 행동의 확률 =', dist.probs.detach())
print('0번을 할 로그확률 =', round(logp.item(), 4))

G_good, G_bad = 10.0, -10.0

for name, G in [('잘 됐을 때 (G=+10)', G_good), ('안 됐을 때 (G=-10)', G_bad)]:
    logits.grad = None
    loss = -(logp * G)                 # ★ 음수가 붙는 이유 ★
    loss.backward(retain_graph=True)
    print(f'\n  {name}')
    print(f'    loss = {loss.item():7.2f}')
    print(f'    0번 행동 점수의 기울기 = {logits.grad[0,0].item():+.3f}')

print('\n  → G가 양수면 그 행동의 점수를 올리는 방향으로,')
print('    음수면 내리는 방향으로 기울기가 생깁니다.')
print('  → 옵티마이저는 "손실을 줄이는" 쪽으로만 움직이므로,')
print('    "점수를 키우고 싶다" = "-점수를 줄이고 싶다" 로 뒤집어 쓴 것입니다.')
print('    그래서 앞에 마이너스가 붙습니다.')
