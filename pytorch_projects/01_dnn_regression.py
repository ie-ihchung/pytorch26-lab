# ============================================================
# [파이토치 응용 ①] DNN 회귀 — 집값 예측
# ------------------------------------------------------------
# 교재 9장에 해당합니다.
#
# 2일차 오전에 배운 학습 5단계가 그대로 나옵니다. 달라진 건 데이터뿐입니다.
#   ① 예측  ② 손실  ③ 기울기 지우기  ④ 역전파  ⑤ 한 걸음
#
# 강화학습과 뭐가 다른가요?
#   여기는 정답이 있습니다. "이 집은 3억" 이라고 알려 줍니다.
#   강화학습은 정답 없이 점수만 있었습니다. 그 차이만 빼면 코드는 같습니다.
#
# 코랩에서: 설치할 것 없습니다. GPU 도 필요 없습니다.
# 걸리는 시간: 1분 안팎
# ============================================================
import torch                                   # 파이토치
import torch.nn as nn                           # 신경망 부품 상자
import numpy as np                              # 숫자 계산
from sklearn.datasets import fetch_california_housing   # 집값 데이터
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0)                            # 결과 고정
np.random.seed(0)

print('=' * 58)
print('1. 데이터 준비 — 캘리포니아 집값')
print('=' * 58)

data = fetch_california_housing()               # 인터넷에서 자동으로 받아 옵니다
X, y = data.data, data.target                   # X = 특징 8개, y = 집값

print(f'  전체 데이터  {X.shape[0]:,}개')
print(f'  특징 개수    {X.shape[1]}개')
print(f'  특징 이름    {", ".join(data.feature_names)}')
print(f'  집값 범위    {y.min():.2f} ~ {y.max():.2f} (10만 달러 단위)')

# 학습용과 시험용으로 나눈다 (8:2)
# 시험용은 학습에 절대 쓰지 않습니다 — 외운 것인지 배운 것인지 구분하려고
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

# 값의 크기를 맞춘다 (표준화)
#   방 개수는 5쯤, 인구는 1000쯤입니다. 크기가 너무 다르면 학습이 잘 안 됩니다.
#   평균 0, 표준편차 1로 맞춰 줍니다.
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr)               # 학습 데이터로 기준을 정하고
X_te = scaler.transform(X_te)                   # 시험 데이터엔 그 기준을 적용만

# 넘파이 -> 텐서 (신경망은 텐서를 받습니다)
X_tr = torch.tensor(X_tr, dtype=torch.float32)
y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)   # (N,) -> (N, 1)
X_te = torch.tensor(X_te, dtype=torch.float32)
y_te = torch.tensor(y_te, dtype=torch.float32).unsqueeze(1)

print(f'\n  학습용 {len(X_tr):,}개 / 시험용 {len(X_te):,}개')
print(f'  X_tr 모양 {tuple(X_tr.shape)}   y_tr 모양 {tuple(y_tr.shape)}')


print()
print('=' * 58)
print('2. 모델 — 층을 세 개 쌓은 신경망')
print('=' * 58)

model = nn.Sequential(
    nn.Linear(8, 64), nn.ReLU(),                # 특징 8개 -> 64개
    nn.Linear(64, 32), nn.ReLU(),               # 64 -> 32
    nn.Linear(32, 1),                           # 32 -> 1 (집값 하나)
)
# 마지막에 활성화를 안 붙입니다. 집값은 어떤 값이든 나올 수 있으니까요.
# (강화학습에서 Q값에 활성화를 안 붙였던 것과 같은 이유입니다)

print(model)
total = sum(p.numel() for p in model.parameters())
print(f'\n  학습할 숫자 개수: {total:,}개')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()                        # 숫자를 맞히는 문제 -> MSE


print()
print('=' * 58)
print('3. 학습 — 그 5단계 그대로')
print('=' * 58)

BATCH = 256                                     # 한 번에 256개씩

for epoch in range(30):
    model.train()                               # 학습 모드
    perm = torch.randperm(len(X_tr))            # 순서를 섞는다 (매 에폭마다)

    for i in range(0, len(X_tr), BATCH):
        idx = perm[i:i + BATCH]                 # 256개 뽑기
        pred = model(X_tr[idx])                 # ① 예측
        loss = criterion(pred, y_tr[idx])       # ② 손실
        optimizer.zero_grad()                   # ③ 기울기 지우기
        loss.backward()                         # ④ 역전파
        optimizer.step()                        # ⑤ 한 걸음

    if epoch % 5 == 0 or epoch == 29:
        model.eval()                            # 평가 모드
        with torch.no_grad():                   # 채점만 할 땐 미분 불필요
            tr_loss = criterion(model(X_tr), y_tr).item()
            te_loss = criterion(model(X_te), y_te).item()
        print(f'  epoch {epoch:2d}   학습 손실 {tr_loss:.4f}   시험 손실 {te_loss:.4f}')

print('''
  두 손실을 나란히 보는 이유
    학습 손실만 줄고 시험 손실이 늘면 = 외운 것입니다 (과적합).
    둘 다 줄면 = 제대로 배운 것입니다.''')


print()
print('=' * 58)
print('4. 실제로 얼마나 맞히나')
print('=' * 58)

model.eval()
with torch.no_grad():
    pred = model(X_te)

print('   실제 집값   예측 집값    차이')
print('  ' + '-' * 34)
for i in range(8):
    a, b = y_te[i].item(), pred[i].item()
    print(f'  {a:8.2f}   {b:8.2f}   {b - a:+7.2f}')

err = (pred - y_te).abs().mean().item()
print(f'\n  평균 오차: {err:.3f} (10만 달러 단위)')
print(f'  = 약 {err * 100000:,.0f} 달러쯤 빗나갑니다')

# ============================================================
# 바꿔 보기
#   1) 층을 하나 더 쌓아 보세요. 좋아지나요?
#   2) epoch 을 100으로 늘리면? 시험 손실이 언제부터 안 줄어드는지 보세요.
#      그 지점이 "더 배울 게 없는" 지점입니다.
#   3) 표준화(StandardScaler)를 빼고 돌려 보세요.
#      학습이 훨씬 안 됩니다 — 값의 크기를 맞추는 게 왜 중요한지 알 수 있습니다.
#   4) lr 을 1e-2, 1e-4 로 바꿔 보세요. 2일차 오전에 본 그 감각입니다.
# ============================================================
