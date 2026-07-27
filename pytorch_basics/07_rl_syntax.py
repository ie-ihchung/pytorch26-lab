# ============================================================
# [파이토치 기초 ⑦] 강화학습 코드에서 실제로 만나는 문법만 모아서
# ------------------------------------------------------------
# 2·3일차 코드를 읽다 "이건 뭐지?" 하고 걸리는 것들만 골랐습니다.
# 이 파일 하나면 수업 코드의 문법은 거의 다 읽힙니다.
#
# 실행: python 07_rl_syntax.py    (2초)
# ============================================================
import torch
import torch.nn as nn

torch.manual_seed(0)

print('=' * 58)
print('1. gather — "내가 실제로 한 행동의 Q값만 뽑아라"')
print('=' * 58)

# 상태 4개를 한 번에 처리했다고 하자. 각 상태마다 행동 3개의 Q값이 나온다.
Q = torch.tensor([[1.0, 5.0, 2.0],
                  [7.0, 0.0, 3.0],
                  [2.0, 2.0, 9.0],
                  [4.0, 6.0, 1.0]])
actions = torch.tensor([1, 0, 2, 1])          # 실제로 한 행동 (상태별로 하나씩)

print('Q (4개 상태 × 행동 3개)\n', Q)
print('실제로 한 행동:', actions.tolist())

a = actions.unsqueeze(1)                      # (4,) → (4, 1) 모양 맞추기
picked = Q.gather(1, a)                       # 1번 축(행동 축)에서 골라 뽑는다

print('\nunsqueeze(1) 후 모양:', a.shape)
print('gather 결과:\n', picked)
print('→ 5, 7, 9, 6. 각 줄에서 지정한 자리 하나씩만 뽑혔습니다.')
print('  ※ gather 없이 Q[range(4), actions] 로도 됩니다. 결과는 같습니다:')
print(' ', Q[range(4), actions].tolist())

print()
print('=' * 58)
print('2. max(1) — "각 줄에서 가장 큰 값"')
print('=' * 58)

vals, idxs = Q.max(1)                         # 1번 축 방향으로 최대
print('최댓값  :', vals.tolist())
print('그 위치 :', idxs.tolist(), ' ← argmax 와 같습니다')
print('\nkeepdim=True 를 주면 모양이 유지됩니다:')
print('  keepdim 없이 :', Q.max(1)[0].shape)
print('  keepdim=True :', Q.max(1, keepdim=True)[0].shape)
print('  ※ 뒤에서 (4,1) 짜리 보상과 더해야 해서 keepdim 을 씁니다.')

print()
print('=' * 58)
print('3. 모양 맞추기 — 이것 때문에 오류가 제일 많이 납니다')
print('=' * 58)

r = torch.tensor([1.0, 1.0, 1.0, 1.0])        # 보상 (4,)
nxt = Q.max(1, keepdim=True)[0]               # 다음 상태 값 (4,1)

wrong = r + 0.99 * nxt.squeeze() * 1          # 둘 다 (4,) → 정상
print('맞게 한 경우 모양:', (r + 0.99 * nxt.squeeze()).shape)

broadcast = r + 0.99 * nxt                    # (4,) 와 (4,1) → (4,4) 로 퍼진다!
print('잘못 섞은 경우 모양:', broadcast.shape, ' ← 4×4 로 부풀었습니다')
print('''
  → 오류가 안 나고 조용히 잘못된 값이 만들어지는 것이 제일 위험합니다.
    손실이 이상하면 무조건 .shape 를 찍어 보세요.''')

print()
print('=' * 58)
print('4. no_grad 와 detach — 정답(과녁)은 고정한다')
print('=' * 58)

net = nn.Linear(3, 1)
s2 = torch.randn(4, 3)

with torch.no_grad():                          # 이 블록 안은 미분 추적 안 함
    target = net(s2)
print('no_grad 로 만든 타깃 requires_grad =', target.requires_grad)

adv = net(s2)
print('그냥 만든 값       requires_grad =', adv.requires_grad)
print('detach() 붙이면    requires_grad =', adv.detach().requires_grad)

print('''
  둘 다 "여기서 끊는다"는 뜻이지만 쓰는 자리가 다릅니다.
    no_grad : 여러 줄을 통째로 끌 때 (타깃 계산 블록 전체)
    detach  : 이미 만들어진 값 하나만 끊을 때 (advantage.detach())

  RL 에서 이걸 빠뜨리면:
    타깃까지 학습되어 과녁이 도망갑니다 → 손실이 0으로 수렴하는데 성능은 그대로''')

print()
print('=' * 58)
print('5. Categorical — 확률대로 행동 뽑기 (Policy Gradient)')
print('=' * 58)

logits = torch.tensor([2.0, 1.0, 0.0])         # 신경망이 낸 점수
dist = torch.distributions.Categorical(logits=logits)

print('점수(logits) =', logits.tolist())
print('확률로 바뀌면 =', [round(v, 3) for v in dist.probs.tolist()])

samples = [int(dist.sample()) for _ in range(1000)]
counts = [samples.count(i) for i in range(3)]
print('1000번 뽑아 보면:', counts, ' → 확률대로 나옵니다')

a = torch.tensor(0)
print('\n0번을 뽑을 로그확률 log_prob(0) =', round(dist.log_prob(a).item(), 4))
print('  (확률 0.665 의 로그 = -0.408)')
print('엔트로피 entropy() =', round(dist.entropy().item(), 4))
print('  → 확률이 골고루일수록 큽니다. "아직 탐험 중"이라는 뜻입니다.')

even = torch.distributions.Categorical(logits=torch.zeros(3))
print('  완전히 균등할 때 =', round(even.entropy().item(), 4), '(= ln 3)')

print()
print('=' * 58)
print('6. Normal — 연속 행동 뽑기 (3일차 DDPG/SAC)')
print('=' * 58)

mu = torch.tensor([0.0])                       # 평균 (어느 쪽으로 갈지)
std = torch.tensor([1.0])                      # 표준편차 (얼마나 흔들지)
nd = torch.distributions.Normal(mu, std)

a = nd.sample()
print('뽑은 행동 =', round(a.item(), 4))
print('log_prob  =', round(nd.log_prob(a).item(), 4))

squashed = torch.tanh(a) * 2.0                 # [-1,1] 로 누르고 범위만큼 곱한다
print('\ntanh 로 누른 뒤 ×2 =', round(squashed.item(), 4))
print('  → Pendulum 의 행동 범위가 [-2, 2] 라서 이렇게 맞춥니다.')

print()
print('=' * 58)
print('7. 넘파이 ↔ 텐서 — 환경과 신경망 사이의 통역')
print('=' * 58)

print('''  gymnasium 환경은 넘파이를 줍니다. 신경망은 텐서를 받습니다.
  그래서 매 스텝 이 변환이 들어갑니다.

    s, _ = env.reset()                                # 넘파이 (4,)
    t = torch.tensor(s, dtype=torch.float32)          # 텐서로
    q = net(t)                                        # 신경망 통과
    a = int(q.argmax())                               # 파이썬 int 로 (env.step 이 요구)
    s2, r, term, trunc, _ = env.step(a)

  자주 나는 오류 두 가지:
    · dtype 안 맞음  → 반드시 dtype=torch.float32
    · int 아님       → env.step(a) 에는 int(a) 로 넣기''')

print()
print('=' * 58)
print('8. 여러 개를 한 번에 (배치) — 리스트를 텐서로 쌓기')
print('=' * 58)

batch = [(torch.randn(4), 1, 1.0), (torch.randn(4), 0, 1.0), (torch.randn(4), 1, 0.0)]
states, acts, rews = zip(*batch)               # 세로로 갈라낸다

S = torch.stack(states)                                    # (3, 4)
A = torch.tensor(acts, dtype=torch.int64).unsqueeze(1)     # (3, 1)
R = torch.tensor(rews, dtype=torch.float32).unsqueeze(1)   # (3, 1)

print('상태 S :', S.shape)
print('행동 A :', A.shape, ' ← int64 여야 gather 가 됩니다')
print('보상 R :', R.shape, ' ← float32')
print('''
  ※ 행동은 int64, 나머지는 float32. 이 규칙만 지키면 대부분 통과합니다.
  ※ torch.stack 은 새 축을 만들며 쌓고, torch.cat 은 기존 축에 이어붙입니다.''')

print()
print('=' * 58)
print('정리 — 이 8가지가 2·3일차 코드의 문법 전부입니다')
print('=' * 58)
print('''  gather        행동별 Q 뽑기
  max/argmax    최선의 행동 찾기
  unsqueeze     모양 맞추기
  no_grad       타깃 계산 블록 끄기
  detach        값 하나만 끊기
  Categorical   이산 행동 뽑기
  Normal+tanh   연속 행동 뽑기
  stack/tensor  여러 경험을 한 덩어리로''')
