# ============================================================
# [파이토치 기초 ⑤] 신경망 만들기 — nn.Module
# ------------------------------------------------------------
# 직선 하나로는 못 푸는 문제가 대부분입니다. 층을 여러 개 쌓습니다.
# 파이토치에서 신경망을 만드는 방법은 크게 두 가지입니다.
#
# 실행: python 05_nn_module.py    (3초)
# ============================================================
import torch
import torch.nn as nn

torch.manual_seed(0)

print('=' * 55)
print('1. 방법 A — nn.Sequential (간단할 때)')
print('=' * 55)

net_a = nn.Sequential(
    nn.Linear(4, 64),      # 4개 받아서 64개로 늘린다
    nn.ReLU(),             # 음수는 0으로 (직선을 구부리는 장치)
    nn.Linear(64, 64),     # 64 → 64
    nn.ReLU(),
    nn.Linear(64, 2),      # 64 → 2  (행동 2개의 Q값)
)
print(net_a)
print('\n  위에서 아래로 순서대로 흘러가기만 하면 이걸로 충분합니다.')

print()
print('=' * 55)
print('2. 방법 B — class 로 직접 만들기 (중간에 뭔가 해야 할 때)')
print('=' * 55)


class QNetwork(nn.Module):
    """
    nn.Module 을 상속받으면 파이토치가 파라미터 관리를 알아서 해줍니다.
    반드시 지켜야 할 두 가지:
      ① __init__ 안에서 super().__init__() 을 먼저 부른다
      ② forward(self, x) 를 정의한다 — 데이터가 흘러가는 길
    """

    def __init__(self, n_obs, n_act):
        super().__init__()                      # ① 이걸 빠뜨리면 오류가 납니다
        self.fc1 = nn.Linear(n_obs, 64)         # 층을 self. 로 두면 파라미터로 등록됩니다
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, n_act)

    def forward(self, x):                       # ② 데이터가 지나가는 길
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)                      # 마지막엔 활성화를 안 씁니다 (Q값은 음수도 가능)


net_b = QNetwork(4, 2)
print(net_b)

print()
print('=' * 55)
print('3. 써 보기 — 한 개 넣기 vs 여러 개 한꺼번에 넣기')
print('=' * 55)

one = torch.randn(4)                    # 상태 하나 (4개 숫자)
print('입력 모양', one.shape, '→ 출력 모양', net_b(one).shape)

many = torch.randn(32, 4)               # 상태 32개를 한 번에
print('입력 모양', many.shape, '→ 출력 모양', net_b(many).shape)
print('  ※ 앞의 32 를 "배치(batch)" 라고 합니다.')
print('    한 번에 여러 개를 처리하는 게 훨씬 빠릅니다. RL 코드는 거의 이 모양입니다.')

print()
print('=' * 55)
print('4. 파라미터 들여다보기')
print('=' * 55)

total = 0
for name, p in net_b.named_parameters():
    print(f'  {name:12s} {str(tuple(p.shape)):12s} 개수 {p.numel():6d}')
    total += p.numel()
print(f'  {"합계":12s} {"":12s} 개수 {total:6d}')
print('  → 이 숫자들을 조금씩 고쳐 가는 것이 "학습" 입니다.')

print()
print('=' * 55)
print('5. train() 과 eval() — 모드 전환')
print('=' * 55)

net_c = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.5), nn.Linear(8, 2))
x = torch.ones(1, 4)

net_c.train()                            # 학습 모드 — Dropout 이 작동한다
outs = [net_c(x).detach().sum().item() for _ in range(3)]
print('train() 모드에서 3번 →', [round(v, 3) for v in outs], ' 값이 매번 다르다')

net_c.eval()                             # 평가 모드 — Dropout 이 꺼진다
outs = [net_c(x).detach().sum().item() for _ in range(3)]
print('eval()  모드에서 3번 →', [round(v, 3) for v in outs], ' 값이 항상 같다')
print('  ※ 우리 강화학습 코드는 Dropout/BatchNorm 을 안 써서 차이가 없지만,')
print('    나중에 이미지 모델을 쓸 땐 평가 전에 eval() 을 꼭 불러야 합니다.')

print()
print('=' * 55)
print('6. 저장하고 다시 불러오기')
print('=' * 55)

torch.save(net_b.state_dict(), 'my_net.pth')     # 숫자들만 저장 (구조는 저장 안 함)
print('저장 완료 → my_net.pth')

net_new = QNetwork(4, 2)                          # 같은 구조를 먼저 만들고
net_new.load_state_dict(torch.load('my_net.pth')) # 숫자를 부어 넣는다
net_new.eval()

same = torch.allclose(net_b(many), net_new(many))
print('불러온 모델이 원본과 같은 값을 내는가 →', same)
print('  ※ 구조(class 정의)는 코드에 있어야 합니다. 파일에는 숫자만 들어갑니다.')

import os
os.remove('my_net.pth')                           # 실습이니 지웁니다

# ============================================================
# 바꿔 보기
#   1) 층을 하나 더 쌓아 보세요 — 파라미터 개수가 얼마나 늘어나나요?
#   2) nn.ReLU() 를 nn.Tanh() 로 바꾸면? (다음 파일 06 에서 다룹니다)
#   3) super().__init__() 을 지워 보세요 → 어떤 오류가 나는지 확인
# ============================================================
