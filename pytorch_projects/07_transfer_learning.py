# ============================================================
# [파이토치 응용 ⑦] 전이학습 — 사진 분류기를 10분 만에
# ------------------------------------------------------------
# 교재 16장(날씨 이미지 분류)의 방식을 코랩에 맞게 재구성했습니다.
#
# ②③에서 CNN 을 처음부터 만들었습니다. 시간도 걸리고 정확도도 한계가 있었죠.
#
# 이미 수백만 장으로 학습된 모델(ResNet)을 가져오면 어떨까요?
#   그 모델은 이미 "선, 모양, 질감"을 압니다.
#   우리는 마지막 판단하는 층 하나만 바꿔 끼웁니다.
#
# ★ 이게 실무에서 가장 많이 쓰는 방식입니다 ★
#   처음부터 만드는 일은 드뭅니다.
#   ⑥번 BERT 도 같은 발상이었습니다 (글 대신 그림일 뿐).
#
# 코랩에서: [런타임] -> [런타임 유형 변경] -> GPU 권장
# 걸리는 시간: GPU 로 5분 / CPU 로 20분
# ============================================================
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, models

torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 장치: {device}')

CLASSES = ['비행기', '자동차', '새', '고양이', '사슴',
           '개', '개구리', '말', '배', '트럭']


print()
print('=' * 58)
print('1. 데이터 — CIFAR-10 (10종 사진)')
print('=' * 58)

# ResNet 은 224x224 컬러 사진으로 학습됐습니다.
# CIFAR-10 은 32x32 이라 크기를 맞춰 줍니다.
transform = transforms.Compose([
    transforms.Resize(224),                     # 32x32 -> 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet 의 평균
                         [0.229, 0.224, 0.225]),  # ImageNet 의 표준편차
])
# 왜 ImageNet 기준으로 표준화하나요?
#   ResNet 이 그 기준으로 학습됐기 때문입니다. 다른 기준을 쓰면 성능이 떨어집니다.

train_set = torchvision.datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=transform)

# 수업용으로 줄입니다. 전체(5만 장)를 쓰면 오래 걸립니다.
train_set = torch.utils.data.Subset(train_set, range(6000))
test_set = torch.utils.data.Subset(test_set, range(1500))

train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=128)

print(f'  학습용 {len(train_set):,}장 / 시험용 {len(test_set):,}장')
print(f'  분류할 종류: {", ".join(CLASSES)}')


print()
print('=' * 58)
print('2. 이미 학습된 ResNet18 가져오기')
print('=' * 58)

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# weights=... 로 '미리 학습된 숫자들'을 함께 받아 옵니다.
# (예전에는 pretrained=True 였는데 torchvision 0.13 부터 이 방식으로 바뀌었습니다)

print(f'  받아온 모델 파라미터 {sum(p.numel() for p in model.parameters()):,}개')
print(f'  원래 출력 개수: {model.fc.out_features}개 (ImageNet 1000종)')

# ── 핵심: 본체는 얼려 두고 마지막 층만 바꾼다 ──
for param in model.parameters():
    param.requires_grad = False              # 전부 얼린다 (학습 안 함)

model.fc = nn.Linear(model.fc.in_features, 10)   # 마지막 층만 새로 (10종으로)
# 새로 만든 층은 requires_grad 가 True 입니다 (기본값)

model = model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\n  실제로 학습할 파라미터: {trainable:,}개')
print(f'  -> 전체의 {trainable / sum(p.numel() for p in model.parameters()) * 100:.2f}% 만 학습합니다')
print('     나머지는 이미 잘 배운 것이라 그대로 씁니다.')

optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
# 옵티마이저에 model.fc 만 넘깁니다. 얼린 부분은 어차피 안 바뀝니다.
criterion = nn.CrossEntropyLoss()


print()
print('=' * 58)
print('3. 학습 — 3 에폭이면 충분합니다')
print('=' * 58)


def evaluate():
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            n += len(y)
    return correct / n


for epoch in range(3):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'  epoch {epoch}   시험 정확도 {evaluate()*100:5.2f}%')

print('''
  3 에폭에 85% 넘으면 정상입니다.

  ②③에서 CNN 을 처음부터 만들었을 때와 비교해 보세요.
  훨씬 적은 에폭으로 더 높은 정확도가 나옵니다.
  -> 이미 배운 것을 가져다 쓰는 것이 그만큼 강력합니다.''')


print()
print('=' * 58)
print('4. 실제로 맞히는지 보기')
print('=' * 58)

model.eval()
x, y = next(iter(test_loader))
x, y = x[:8].to(device), y[:8]
with torch.no_grad():
    pred = model(x).argmax(1).cpu()

for i in range(8):
    mark = 'O' if pred[i] == y[i] else 'X'
    print(f'  [{mark}] 정답 {CLASSES[y[i]]:5s} -> 예측 {CLASSES[pred[i]]}')

# ============================================================
# 여러분 사진으로 바꾸는 법 (교재 16장 날씨 분류처럼)
# ------------------------------------------------------------
#   폴더를 이렇게 만들어 두면 그대로 됩니다.
#
#     my_data/
#       train/
#         맑음/  sunny001.jpg  sunny002.jpg ...
#         비/    rain001.jpg ...
#         눈/    snow001.jpg ...
#       test/
#         맑음/ ...
#
#   코드에서 데이터 부분만 바꾸면 됩니다:
#
#     train_set = torchvision.datasets.ImageFolder('my_data/train', transform=transform)
#     test_set  = torchvision.datasets.ImageFolder('my_data/test',  transform=transform)
#     model.fc  = nn.Linear(model.fc.in_features, len(train_set.classes))
#
#   코랩에서는 왼쪽 폴더 아이콘으로 올리거나
#   구글 드라이브를 연결해서 쓰시면 됩니다.
#
# 바꿔 보기
#   1) 얼리지 말고 전부 학습시켜 보세요 (requires_grad = True).
#      느려지는데 정확도는 얼마나 오르나요?
#   2) resnet18 을 resnet50 으로 바꿔 보세요. 더 크고 더 정확합니다.
#   3) epoch 을 10으로 늘리면 얼마나 더 오르나요? 한계가 보입니다.
# ============================================================
