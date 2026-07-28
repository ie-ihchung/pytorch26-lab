# ============================================================
# [파이토치 응용 ⑤] GAN — 없는 숫자를 만들어 내기
# ------------------------------------------------------------
# 교재 13장에 해당합니다.
#
# 지금까지는 '알아맞히는' 일이었습니다. 이제 '만들어 냅니다'.
#
# 위조지폐범과 경찰 이야기입니다.
#   생성자(Generator)   가짜 숫자 그림을 만든다        — 위조지폐범
#   판별자(Discriminator) 진짜인지 가짜인지 가려낸다   — 경찰
#
# 둘이 계속 겨룹니다.
#   경찰이 잘 잡으면 위조범이 더 정교해지고,
#   위조범이 잘 속이면 경찰이 더 꼼꼼해집니다.
#   결국 사람 눈에도 진짜 같은 그림이 나옵니다.
#
# ★ 강화학습과 닮은 점 ★
#   정답을 주지 않습니다. 상대의 반응이 곧 점수입니다.
#   강화학습에서 환경이 점수를 주듯, 여기서는 판별자가 점수를 줍니다.
#
# 코랩에서: [런타임] -> [런타임 유형 변경] -> GPU 를 켜세요.
# 걸리는 시간: GPU 로 5분 / CPU 로 25분 (EPOCHS 를 줄이셔도 됩니다)
# ============================================================
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

torch.manual_seed(0)

# GPU 가 있으면 쓰고 없으면 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'사용 장치: {device}')
if device.type == 'cpu':
    print('  ※ CPU 입니다. 오래 걸리니 EPOCHS 를 20 정도로 줄이셔도 됩니다.')

LATENT = 64        # 생성자에게 주는 '씨앗' 숫자의 개수
BATCH = 128        # 한 번에 처리할 그림 수
EPOCHS = 50        # 몇 바퀴 학습할지


print()
print('=' * 58)
print('1. 데이터 — MNIST 손글씨 숫자')
print('=' * 58)

# 그림 값을 -1 ~ 1 로 맞춥니다.
# 왜? 생성자 마지막에 Tanh 를 쓸 건데 Tanh 출력이 -1~1 이라서요.
# (3일차에서 행동 범위를 Tanh 로 맞춘 것과 같은 이야기입니다)
transform = transforms.Compose([
    transforms.ToTensor(),                      # 0~255 -> 0~1
    transforms.Normalize((0.5,), (0.5,)),       # 0~1 -> -1~1
])

train_set = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH, shuffle=True)

print(f'  그림 {len(train_set):,}장, 크기 28x28')


print()
print('=' * 58)
print('2. 두 신경망 만들기')
print('=' * 58)


class Generator(nn.Module):
    """씨앗 숫자 64개 -> 28x28 그림 (위조지폐범)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 28 * 28),
            nn.Tanh(),                          # -1 ~ 1 (데이터와 범위를 맞춤)
        )
        # LeakyReLU 를 쓰는 이유: ReLU 는 음수를 전부 0으로 죽입니다.
        # GAN 은 기울기가 죽으면 학습이 멈춰서, 음수도 조금 통과시킵니다.

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)  # 한 줄 -> 그림 모양으로


class Discriminator(nn.Module):
    """28x28 그림 -> 진짜일 확률 (경찰)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                       # 그림 -> 한 줄로 펴기
            nn.Linear(28 * 28, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),                       # 0~1 = 진짜일 확률
        )

    def forward(self, x):
        return self.net(x)


G = Generator().to(device)
D = Discriminator().to(device)

print(f'  생성자 파라미터 {sum(p.numel() for p in G.parameters()):,}개')
print(f'  판별자 파라미터 {sum(p.numel() for p in D.parameters()):,}개')

# 옵티마이저를 따로 둡니다. 서로를 건드리지 않게 하려고요.
# (3일차 DDPG 에서 배우와 평론가를 따로 둔 것과 같습니다)
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

criterion = nn.BCELoss()                        # 진짜/가짜 두 갈래 -> 이진 분류 손실


print()
print('=' * 58)
print('3. 학습 — 경찰과 위조범이 번갈아 배웁니다')
print('=' * 58)

# 학습 과정을 비교하려고 고정된 씨앗을 하나 준비합니다.
# 매 에폭 같은 씨앗으로 그림을 만들면 변해 가는 과정이 보입니다.
fixed_z = torch.randn(16, LATENT, device=device)
snapshots = []

for epoch in range(EPOCHS):
    d_sum, g_sum, n = 0.0, 0.0, 0

    for real, _ in loader:                      # 라벨(_)은 안 씁니다. 숫자가 뭔지는 관심 없음
        real = real.to(device)
        bs = real.size(0)

        # 정답표: 진짜는 1, 가짜는 0
        ones = torch.ones(bs, 1, device=device)
        zeros = torch.zeros(bs, 1, device=device)

        # ── ① 경찰(판별자) 훈련 ──
        z = torch.randn(bs, LATENT, device=device)   # 씨앗 뽑기
        fake = G(z)                                   # 위조범이 가짜를 만든다

        d_real = criterion(D(real), ones)             # 진짜를 진짜라고 해야 함
        d_fake = criterion(D(fake.detach()), zeros)   # 가짜를 가짜라고 해야 함
        # ★ fake.detach() ★ 여기가 중요합니다.
        #   경찰을 훈련시키는 중이니 위조범까지 고쳐지면 안 됩니다.
        #   2일차 Actor-Critic 에서 advantage.detach() 를 쓴 것과 같은 이유입니다.
        d_loss = d_real + d_fake

        opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        # ── ② 위조범(생성자) 훈련 ──
        z = torch.randn(bs, LATENT, device=device)
        fake = G(z)
        g_loss = criterion(D(fake), ones)
        # 위조범의 목표: 경찰이 '진짜(1)'라고 말하게 만드는 것
        # 여기서는 detach 를 쓰지 않습니다 — 위조범까지 기울기가 가야 하니까요.

        opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        d_sum += d_loss.item(); g_sum += g_loss.item(); n += 1

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f'  epoch {epoch:2d}   경찰 손실 {d_sum/n:.4f}   위조범 손실 {g_sum/n:.4f}')
        with torch.no_grad():
            snapshots.append((epoch, G(fixed_z).cpu()))

print('''
  ★ 손실로 판단하면 안 됩니다 ★
    둘이 겨루는 구조라 한쪽이 잘하면 다른 쪽 손실이 올라갑니다.
    손실이 줄어드는 것이 좋은 것이 아닙니다. 균형을 이루는 것이 좋은 것입니다.
    눈으로 그림을 봐야 합니다.

    (강화학습에서 손실 대신 점수를 봤던 것과 같은 이야기입니다)''')


print()
print('=' * 58)
print('4. 만들어진 그림 보기')
print('=' * 58)

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(snapshots), 8, figsize=(10, 1.4 * len(snapshots)))
    if len(snapshots) == 1:
        axes = axes.reshape(1, -1)

    for r, (ep, imgs) in enumerate(snapshots):
        for c in range(8):
            ax = axes[r][c]
            ax.imshow(imgs[c][0], cmap='gray')
            ax.axis('off')
            if c == 0:
                ax.set_title(f'epoch {ep}', fontsize=9, loc='left')

    plt.tight_layout()
    plt.show()
    print('  위에서 아래로 내려오면서 숫자 모양이 잡혀 갑니다.')
except ImportError:
    print('  (matplotlib 이 없어 그림은 건너뜁니다)')

# ============================================================
# 바꿔 보기
#   1) EPOCHS 를 100 으로 늘리면 더 선명해집니다.
#   2) LATENT 를 16 으로 줄이면? 만들 수 있는 그림의 다양성이 줄어듭니다.
#   3) 생성자만 학습률을 두 배로 해보세요. 균형이 깨지면 어떻게 되는지 봅니다.
#      (한쪽이 너무 세지면 학습이 무너집니다 — GAN 의 가장 흔한 실패)
#   4) fake.detach() 를 빼고 돌려 보세요.
#      경찰을 훈련시키면서 위조범까지 망가집니다. 2일차에서 본 그 문제입니다.
# ============================================================
