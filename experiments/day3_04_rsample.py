# ============================================================
# [3일차 4교시 실습] rsample 과 tanh 보정 — 오늘 가장 어려운 두 곳
# ------------------------------------------------------------
# SAC 코드에서 사람들이 가장 많이 틀리는 두 곳을 눈으로 봅니다.
#   ① sample() 대신 rsample() 을 써야 하는 이유
#   ② tanh 보정 항을 빼먹으면 무슨 일이 나는가
#
# 둘 다 "오류가 안 나고 조용히 잘못되는" 종류라 더 위험합니다.
#
# 실행 시간: 3초
# ============================================================
import torch

torch.manual_seed(0)                       # 결과 고정


print('=' * 60)
print('1. sample() 과 rsample() — 미분이 통과하느냐')
print('=' * 60)

mu = torch.tensor([0.5], requires_grad=True)      # 평균 (학습 대상)
std = torch.tensor([1.0])                          # 흔들림 크기

# ── sample() 로 뽑아 보기 ──
dist = torch.distributions.Normal(mu, std)
a1 = dist.sample()                                 # 그냥 뽑는다
print('sample()  로 뽑은 값 :', round(a1.item(), 4))
print('  requires_grad =', a1.requires_grad, ' ← 미분 연결이 없습니다')

# ── rsample() 로 뽑아 보기 ──
a2 = dist.rsample()                                # 재매개변수화로 뽑는다
print('\nrsample() 로 뽑은 값 :', round(a2.item(), 4))
print('  requires_grad =', a2.requires_grad, ' ← 미분이 연결돼 있습니다')

print('''
  rsample 이 하는 일은 이겁니다:
      a = mu + std * (표준정규에서 뽑은 무작위)
  무작위는 미분과 상관없는 상수로 두고, mu 와 std 만 남깁니다.
  그래서 "mu 를 어느 쪽으로 옮길지" 를 배울 수 있습니다.''')

# ── 실제로 학습이 되는지 확인 ──
print('\n  실제로 학습이 되는지 확인해 봅니다.')
print('  목표: 행동이 2.0 에 가까워지도록 mu 를 옮기기\n')

for name, use_rsample in [('sample()  사용', False), ('rsample() 사용', True)]:
    mu = torch.tensor([0.0], requires_grad=True)   # 항상 0에서 출발
    opt = torch.optim.SGD([mu], lr=0.1)

    for _ in range(50):
        d = torch.distributions.Normal(mu, torch.tensor([1.0]))
        a = d.rsample() if use_rsample else d.sample()
        loss = (a - 2.0) ** 2                      # 2.0 에 가까워지게
        opt.zero_grad()
        try:
            loss.backward()
            opt.step()
        except RuntimeError as e:
            print(f'  {name}: 오류 — {str(e)[:52]}')
            break
    else:
        print(f'  {name}: 50번 학습 후 mu = {mu.item():.4f}')

print('''
  → sample() 쪽은 backward 자체가 안 됩니다 (연결이 끊겨서).
  → rsample() 쪽은 mu 가 2.0 쪽으로 이동합니다.
  ※ SAC 코드에서 sample() 을 쓰면 배우가 전혀 학습되지 않습니다.''')


print()
print('=' * 60)
print('2. tanh 보정 — 빼먹으면 확률이 틀어집니다')
print('=' * 60)

print('''  왜 보정이 필요한가 (비유)
    고무줄에 눈금을 그려 놓고 잡아당기면 눈금 간격이 달라집니다.
    tanh 는 값을 눌러 붙이는 일이라, 눌린 곳은 확률이 촘촘해집니다.
    그 변화만큼 빼 주는 것이 보정 항입니다.
''')

mu = torch.tensor([0.0])
std = torch.tensor([1.0])
dist = torch.distributions.Normal(mu, std)

# 여러 값을 뽑아 두 방식의 확률을 비교합니다
print('   뽑은 값 u    tanh(u)=a    보정 안 함    보정 함     차이')
print('  ' + '-' * 62)
for u_val in [-2.0, -1.0, 0.0, 1.0, 2.0]:
    u = torch.tensor([u_val])
    a = torch.tanh(u)

    logp_raw = dist.log_prob(u).sum(-1)                          # 보정 안 한 것
    logp_fix = logp_raw - torch.log(1 - a.pow(2) + 1e-6).sum(-1) # 보정 한 것

    print(f'  {u_val:8.2f}   {a.item():9.4f}   {logp_raw.item():10.4f}'
          f'  {logp_fix.item():10.4f}  {(logp_fix - logp_raw).item():8.4f}')

print('''
  → 값이 ±1 에 가까워질수록 차이가 커집니다.
    tanh 가 그 근처에서 가장 심하게 눌러 붙이기 때문입니다.
  → 이 보정을 빼먹으면 "엔트로피를 실제보다 크게" 잡습니다.
    그러면 SAC 가 필요 이상으로 탐험만 하고 수렴하지 않습니다.''')


print()
print('=' * 60)
print('3. 확률의 총합으로 검산해 보기')
print('=' * 60)

# 제대로 보정했다면, 변환된 행동의 확률을 다 더하면 1이 되어야 합니다.
# 수치적분으로 확인합니다.
n = 200_000
u = torch.distributions.Normal(0.0, 1.0).sample((n,))
a = torch.tanh(u)

# a 가 [-0.5, 0.5] 안에 들어올 실제 비율
inside = ((a > -0.5) & (a < 0.5)).float().mean()
print(f'  표본으로 센 비율          : {inside.item():.4f}')

# 보정된 확률밀도를 적분해서 같은 구간의 확률 구하기
grid = torch.linspace(-0.499, 0.499, 20000)          # a 격자
u_grid = torch.atanh(grid)                            # 대응하는 u
logp = torch.distributions.Normal(0.0, 1.0).log_prob(u_grid)
logp = logp - torch.log(1 - grid.pow(2) + 1e-12)      # 보정 (밀도 변환)
prob = torch.exp(logp)
integral = torch.trapz(prob, grid)                    # 사다리꼴 적분

print(f'  보정한 밀도를 적분한 값   : {integral.item():.4f}')
print('  → 두 값이 거의 같으면 보정이 맞게 된 것입니다.')

# 보정을 빼면 어떻게 되는지도 확인
logp_no = torch.distributions.Normal(0.0, 1.0).log_prob(u_grid)
integral_no = torch.trapz(torch.exp(logp_no), grid)
print(f'  보정 안 하고 적분한 값    : {integral_no.item():.4f}  ← 안 맞습니다')

# ============================================================
# 바꿔 보기
#   1) 2번 표에서 u 를 ±3, ±4 로 넣어 보세요. 차이가 더 커집니다.
#   2) 1e-6 을 지우고 u = ±5 를 넣어 보세요.
#      a 가 ±1 에 너무 가까워져 log(0) 이 되고 -inf 가 나옵니다.
#      그 안전장치가 왜 필요한지 알 수 있습니다.
#   3) std 를 0.1 로 줄이면 보정 차이가 어떻게 되나요?
# ============================================================
