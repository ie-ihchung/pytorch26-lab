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

import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── 1. Tensor & Autograd ──
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x          # y = x² + 3x
y.backward()
print(x.grad)               # dy/dx = 2x + 3 = 7

# ── 2. nn.Module로 Q-네트워크 정의 ──
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
    def forward(self, s):
        return self.net(s)      # 각 행동의 Q값 벡터

# ── 3. 학습 루프 5단계 (회귀 예제로 패턴 익히기) ──
model = QNetwork(4, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

states = torch.randn(64, 4, device=device)       # 가짜 배치
targets = torch.randn(64, 2, device=device)

for step in range(200):
    pred = model(states)             # ① 순전파
    loss = criterion(pred, targets)  # ② 손실
    optimizer.zero_grad()            # ③ 기울기 초기화
    loss.backward()                  # ④ 역전파
    optimizer.step()                 # ⑤ 갱신
    if step % 50 == 0:
        print(f"step {step:3d}  loss = {loss.item():.4f}")
