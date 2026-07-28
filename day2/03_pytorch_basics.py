# ==========================================================
# 2일차 3교시 — PyTorch 소개 및 구현
# 2026-07-28 (화) 11:30 ~ 12:30 · Value-based & Policy-based Methods
# 원본 파일명: pytorch_basics.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/2#s3
# ==========================================================
# [학습목표]
#  - Tensor, Autograd, nn.Module, Optimizer의 역할을 이해한다
#  - PyTorch 학습 루프의 5단계 정형 패턴을 몸에 익힌다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day2_전체.ipynb를 위에서부터 실행하세요.

import torch                                    # 파이토치 본체
import torch.nn as nn                            # 신경망 부품 상자 (층, 손실함수 등)

# 그래픽카드(GPU)가 있으면 쓰고, 없으면 CPU 를 쓴다.
# 오늘 코드는 아주 작아서 CPU 로도 충분합니다 — 없다고 걱정하지 마세요.
device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. 텐서와 자동미분 — 파이토치가 미분을 대신 해준다
# ============================================================

# requires_grad=True = "이 값에 대해 미분할 거야" 라고 표시하는 것
x = torch.tensor([2.0], requires_grad=True)

y = x ** 2 + 3 * x          # y = x제곱 + 3x  (x=2 이므로 y = 4 + 6 = 10)

y.backward()                # 여기서 미분이 일어난다 -> x.grad 에 답이 들어감

print(x.grad)               # dy/dx = 2x + 3 = 2*2 + 3 = 7
                            # 손으로 계산한 값과 똑같습니다. 파이토치가 대신 해준 것뿐입니다.


# ============================================================
# 2. nn.Module 로 Q-네트워크 만들기
#    상황을 받아서 "각 행동이 얼마나 좋은지"를 내놓는 신경망
# ============================================================

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        # state_dim  = 상황을 나타내는 숫자가 몇 개인지 (CartPole 은 4개)
        # action_dim = 할 수 있는 행동이 몇 가지인지 (CartPole 은 왼/오 2가지)
        # hidden     = 가운데 층의 크기. 클수록 똑똑하지만 느립니다.
        super().__init__()                       # 부모(nn.Module) 준비 — 빠뜨리면 오류

        # nn.Sequential = 위에서 아래로 순서대로 통과시키는 통로
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),   # 4개 -> 128개, 그리고 구부리기
            nn.Linear(hidden, hidden), nn.ReLU(),      # 128개 -> 128개, 또 구부리기
            nn.Linear(hidden, action_dim),             # 128개 -> 2개 (행동별 Q값)
        )
        # 마지막에 활성화 함수를 안 붙이는 이유:
        #   Q값은 -100 일 수도 +500 일 수도 있습니다.
        #   Sigmoid 같은 걸 붙이면 0~1 로 눌려서 표현을 못 합니다.

    def forward(self, s):
        # 데이터가 지나가는 길. model(s) 라고 쓰면 이 함수가 불립니다.
        return self.net(s)                       # 각 행동의 Q값이 한 줄로 나온다


# ============================================================
# 3. 학습 루프 5단계 — 앞으로 모든 코드에 그대로 반복됩니다
# ============================================================

model = QNetwork(4, 2).to(device)                # 신경망을 만들고 CPU/GPU 로 보낸다
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#           ^^^^^^^^^^^^^^^^ 고칠 대상(신경망 속 숫자들)을 알려 준다
#                                                lr = 한 번에 얼마나 움직일지
criterion = nn.MSELoss()                         # 얼마나 틀렸는지 재는 자 (평균제곱오차)

# 진짜 게임 대신 아무 숫자나 만들어 연습합니다 (패턴만 익히는 게 목적)
states = torch.randn(64, 4, device=device)       # 상황 64개, 각각 숫자 4개
targets = torch.randn(64, 2, device=device)      # 정답 64개, 각각 숫자 2개

for step in range(200):                          # 200번 반복해서 배운다
    pred = model(states)                         # ① 예측한다
    loss = criterion(pred, targets)              # ② 얼마나 틀렸나 잰다
    optimizer.zero_grad()                        # ③ 지난 기울기를 지운다 (안 지우면 쌓임!)
    loss.backward()                              # ④ 어디를 고칠지 계산한다
    optimizer.step()                             # ⑤ 그 방향으로 한 걸음 간다

    if step % 50 == 0:                           # 50번마다 한 번씩만 출력
        print(f"step {step:3d}  loss = {loss.item():.4f}")
        # .item() = 텐서 안의 숫자 하나를 꺼내는 것 (그냥 출력하면 tensor(...) 로 나옴)

# 정답이 아무 숫자라서 손실이 0까지 내려가진 않습니다.
# 여기서 볼 것은 "숫자가 줄어드는가" 하나뿐입니다.
