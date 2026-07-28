# ============================================================
# [파이토치 응용 ②] CNN — 손글씨 숫자 알아맞히기
# ------------------------------------------------------------
# 교재 10장에 해당합니다.
#
# 지금까지 신경망에 넣은 것은 '숫자 몇 개'였습니다.
# 이제 '그림'을 넣습니다. 28x28 픽셀짜리 손글씨 숫자입니다.
#
# 왜 CNN 이 따로 필요한가요?
#   그림을 한 줄로 펴서 넣으면 옆 픽셀끼리 붙어 있다는 정보가 사라집니다.
#   CNN 은 작은 창(필터)을 훑어 가며 '모양'을 찾습니다.
#
# 코랩에서: 설치할 것 없습니다. GPU 없어도 됩니다(있으면 더 빠름).
# 걸리는 시간: CPU 로 3분 / GPU 로 1분
# ============================================================
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 장치: {device}')


print()
print('=' * 58)
print('1. 데이터 — MNIST 손글씨')
print('=' * 58)

transform = transforms.Compose([
    transforms.ToTensor(),                          # 0~255 -> 0~1 텐서
    transforms.Normalize((0.1307,), (0.3081,)),     # MNIST 의 평균/표준편차로 표준화
])

train_set = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256)

print(f'  학습용 {len(train_set):,}장 / 시험용 {len(test_set):,}장')
img, label = train_set[0]
print(f'  그림 한 장 모양 {tuple(img.shape)}   (채널, 세로, 가로)')
print('  채널이 1인 이유: 흑백이라서. 컬러면 3(RGB)입니다.')


print()
print('=' * 58)
print('2. CNN 만들기')
print('=' * 58)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Conv2d(들어오는 채널, 나가는 채널, 창 크기)
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),   # 1채널 -> 32채널, 3x3 창
            nn.MaxPool2d(2),                              # 크기 절반으로 (28 -> 14)
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),   # 32 -> 64채널
            nn.MaxPool2d(2),                              # 14 -> 7
        )
        # Conv2d 가 하는 일: 3x3 짜리 작은 창으로 그림 전체를 훑으며
        #   "여기 세로선이 있나?" "여기 곡선이 있나?" 를 찾습니다.
        #   그 창의 내용(필터)을 학습으로 알아냅니다.
        # MaxPool2d: 2x2 안에서 가장 큰 값만 남깁니다.
        #   크기를 줄여 계산을 아끼고, 위치가 조금 달라도 같게 보게 합니다.

        self.classifier = nn.Sequential(
            nn.Flatten(),                   # 64채널 x 7 x 7 -> 한 줄로 펴기
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3),                # 학습 중 30% 를 무작위로 끈다 (외우기 방지)
            nn.Linear(128, 10),             # 숫자 10개 중 하나
        )

    def forward(self, x):
        return self.classifier(self.features(x))


model = CNN().to(device)
print(model)
print(f'\n  학습할 숫자 {sum(p.numel() for p in model.parameters()):,}개')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()           # 10개 중 하나 고르기 -> 교차 엔트로피


print()
print('=' * 58)
print('3. 학습')
print('=' * 58)


def evaluate(loader):
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            n += len(y)
    return correct / n


for epoch in range(5):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)       # ①② 예측 + 손실
        optimizer.zero_grad()               # ③ 기울기 지우기
        loss.backward()                     # ④ 역전파
        optimizer.step()                    # ⑤ 한 걸음

    acc = evaluate(test_loader)
    print(f'  epoch {epoch}   시험 정확도 {acc*100:5.2f}%')

print('\n  98% 넘으면 잘 된 것입니다.')


print()
print('=' * 58)
print('4. 틀린 것 들여다보기')
print('=' * 58)

model.eval()
wrong = []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        bad = (pred != y).nonzero(as_tuple=True)[0]
        for i in bad[:8]:
            wrong.append((x[i].cpu(), y[i].item(), pred[i].item()))
        if len(wrong) >= 8:
            break

print(f'  틀린 것 {len(wrong)}개를 골라 봤습니다')
for _, t, p in wrong:
    print(f'    정답 {t} -> 예측 {p}')

try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(wrong), figsize=(1.4 * len(wrong), 2))
    for ax, (im, t, p) in zip(axes, wrong):
        ax.imshow(im[0], cmap='gray'); ax.axis('off')
        ax.set_title(f'{t}->{p}', fontsize=9)
    plt.tight_layout(); plt.show()
    print('\n  사람이 봐도 헷갈리는 글씨인지 확인해 보세요.')
except ImportError:
    pass

# ============================================================
# 바꿔 보기
#   1) Conv 층을 하나 더 쌓아 보세요. 정확도가 오르나요?
#   2) Dropout(0.3) 을 빼 보세요. 학습 정확도는 오르는데 시험은? (과적합)
#   3) MaxPool 을 빼면 어떻게 되나요? 파라미터가 확 늘고 느려집니다.
#   4) 다음 프로젝트(③)에서 이 코드를 그대로 옷 사진에 써 봅니다.
# ============================================================
