# ============================================================
# [파이토치 응용 ④] RNN — 순서가 있는 데이터 다루기
# ------------------------------------------------------------
# 교재 12장에 해당합니다.
#
# CNN 은 그림을 봤습니다. RNN 은 '순서'를 봅니다.
#
# 여기서는 MNIST 이미지를 위에서 아래로 한 줄씩 읽는 순서 데이터로 봅니다.
#   28줄을 차례로 넣으면서 "지금까지 본 것"을 기억해 나갑니다.
#
# ★ 강화학습과 닮은 점 ★
#   둘 다 시간 순서를 다룹니다.
#   RNN 이 "앞 글자를 기억"한다면, 강화학습은 "앞 상태의 가치를 기억"합니다.
#
# 코랩에서: 설치할 것 없습니다.
# 걸리는 시간: CPU 로 4분 / GPU 로 2분
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
print('1. 그림을 순서 데이터로 보기')
print('=' * 58)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_set = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=256)

print('''  28x28 그림을 이렇게 봅니다.

    1번째 줄 (가로 28칸)  ->  RNN 에 넣고 기억
    2번째 줄 (가로 28칸)  ->  넣고 기억 (1번 줄 기억과 합쳐서)
    ...
    28번째 줄             ->  넣고, 그동안의 기억으로 숫자를 맞힌다

  즉 "길이 28짜리 순서 데이터, 각 시점에 값 28개" 로 봅니다.''')


print()
print('=' * 58)
print('2. RNN 만들기 (LSTM)')
print('=' * 58)


class RNNClassifier(nn.Module):
    def __init__(self, input_size=28, hidden=128, layers=2, classes=10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,     # 한 시점에 들어오는 값 개수 (가로 28칸)
            hidden_size=hidden,        # 기억 상자의 크기
            num_layers=layers,         # LSTM 을 몇 층 쌓을지
            batch_first=True,          # 입력 모양을 (배치, 길이, 값) 으로
        )
        # 왜 그냥 RNN 이 아니라 LSTM 인가요?
        #   기본 RNN 은 앞쪽 기억이 금방 사라집니다 (기울기 소실).
        #   LSTM 은 "기억 상자"를 따로 둬서 오래 기억합니다.

        self.fc = nn.Linear(hidden, classes)   # 마지막 기억 -> 숫자 10개

    def forward(self, x):
        x = x.squeeze(1)               # (배치, 1, 28, 28) -> (배치, 28, 28)
                                       #   채널 차원을 없앤다. 이제 28줄짜리 순서.
        out, _ = self.lstm(x)          # out: 각 시점의 출력 (배치, 28, hidden)
        return self.fc(out[:, -1, :])  # 마지막 시점의 것만 쓴다
        # 왜 마지막만? 28줄을 다 본 뒤의 기억에 전체 정보가 들어 있으니까요.


model = RNNClassifier().to(device)
print(model)
print(f'\n  학습할 숫자 {sum(p.numel() for p in model.parameters()):,}개')

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


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
        loss = criterion(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'  epoch {epoch}   시험 정확도 {evaluate(test_loader)*100:5.2f}%')

print('''
  95% 넘으면 정상입니다. ②번 CNN(98%)보다 조금 낮습니다.

  -> 이미지에는 CNN 이 맞고, RNN 은 글이나 시계열에 맞습니다.
     도구를 문제에 맞게 골라야 한다는 이야기입니다.''')


print()
print('=' * 58)
print('4. 몇 줄까지 보면 맞힐 수 있나')
print('=' * 58)

# RNN 은 순서대로 읽으므로 "중간까지만 보고" 맞혀 볼 수도 있습니다.
model.eval()
x, y = next(iter(test_loader))
x, y = x[:200].to(device), y[:200].to(device)

with torch.no_grad():
    seq = x.squeeze(1)
    for rows in [7, 14, 21, 28]:
        out, _ = model.lstm(seq[:, :rows, :])       # 위에서 rows 줄만 넣는다
        pred = model.fc(out[:, -1, :]).argmax(1)
        acc = (pred == y).float().mean().item()
        print(f'  위에서 {rows:2d}줄만 보면  정확도 {acc*100:5.1f}%')

print('''
  -> 절반쯤 봐도 어느 정도는 맞힙니다.
     사람도 숫자 윗부분만 보고 짐작하는 것과 비슷합니다.''')

# ============================================================
# 바꿔 보기
#   1) nn.LSTM 을 nn.RNN 으로 바꿔 보세요. 정확도가 떨어집니다 (기억이 짧아서).
#   2) nn.GRU 로도 바꿔 보세요. LSTM 과 비슷하면서 더 가볍습니다.
#   3) num_layers 를 1로 줄이면? 3으로 늘리면?
#   4) 그림을 세로로 읽게 바꿔 보세요 (x.squeeze(1).transpose(1,2)).
#      정확도가 달라지나요?
# ============================================================
