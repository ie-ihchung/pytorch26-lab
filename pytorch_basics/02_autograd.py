# ============================================================
# [파이토치 기초 ②] 자동미분 — 파이토치를 쓰는 진짜 이유
# ------------------------------------------------------------
# 신경망 학습은 결국 "어느 쪽으로 조금 움직이면 오차가 줄어드는가"를
# 계속 찾는 일입니다. 그 방향이 미분(기울기, gradient)입니다.
#
# 파이토치의 핵심 기능은 이겁니다:
#   내가 계산식을 쓰기만 하면, 미분은 파이토치가 알아서 해준다.
#
# 실행: python 02_autograd.py    (1초)
# ============================================================
import torch

print('=' * 55)
print('1. 미분을 손으로 확인해 보기')
print('=' * 55)

# requires_grad=True → "이 값에 대해 미분할 거야" 라고 표시하는 것
x = torch.tensor(3.0, requires_grad=True)

y = x ** 2                      # y = x²
print('x =', x.item(), ' y = x² =', y.item())

y.backward()                    # ★ 여기서 미분이 일어난다 ★
                                #   dy/dx 를 계산해 x.grad 에 넣어 준다

print('x.grad =', x.grad.item(), '  ← dy/dx = 2x = 2×3 = 6')
print('  손으로 계산한 값과 같습니다. 파이토치가 대신 해준 것뿐입니다.')

print()
print('=' * 55)
print('2. 조금 복잡해져도 똑같습니다')
print('=' * 55)

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(5.0, requires_grad=True)

z = a * b + a ** 3              # z = ab + a³
z.backward()

print('z = a·b + a³   (a=2, b=5)')
print('  dz/da =', a.grad.item(), '  ← b + 3a² = 5 + 12 = 17')
print('  dz/db =', b.grad.item(), '  ← a = 2')
print('  식이 아무리 길어져도 backward() 한 줄이면 끝납니다.')

print()
print('=' * 55)
print('3. 기울기는 쌓입니다 — zero_grad() 가 필요한 이유')
print('=' * 55)

w = torch.tensor(1.0, requires_grad=True)

for i in range(3):
    out = w * 2                 # d(out)/dw = 2
    out.backward()
    print(f'  {i+1}번째 backward 후  w.grad = {w.grad.item()}')

print('  → 2, 4, 6 으로 계속 더해집니다! 지우지 않으면 누적됩니다.')
w.grad = None                   # 또는 optimizer.zero_grad()
out = w * 2; out.backward()
print(f'  지우고 다시 하면    w.grad = {w.grad.item()}')
print('  ※ 학습 루프에서 opt.zero_grad() 를 빠뜨리면 여기서 사고가 납니다.')

print()
print('=' * 55)
print('4. no_grad — "이건 계산만 하고 미분은 하지 마"')
print('=' * 55)

p = torch.tensor(3.0, requires_grad=True)

q1 = p * 2
print('그냥 계산      : requires_grad =', q1.requires_grad, '  (미분 준비를 하고 있다)')

with torch.no_grad():           # 이 안에서는 미분 준비를 안 한다
    q2 = p * 2
print('no_grad 안에서 : requires_grad =', q2.requires_grad, ' (준비 안 함 → 빠르고 메모리 절약)')
print('  ※ 강화학습에서 "정답(타깃) 만들 때"는 반드시 no_grad 안에서 합니다.')

print()
print('=' * 55)
print('5. detach — "이 값은 여기서 끊는다"')
print('=' * 55)

m = torch.tensor(3.0, requires_grad=True)
n = m * 2

n_detached = n.detach()         # 값은 같지만 연결이 끊긴 복사본
print('n          requires_grad =', n.requires_grad)
print('n.detach() requires_grad =', n_detached.requires_grad)

# 실제로 무슨 차이가 나는지 확인
m.grad = None
(n * 5).backward()              # 연결되어 있으므로 m 까지 미분이 흘러간다
print('\n연결된 채로 backward → m.grad =', m.grad.item(), ' (미분이 m까지 흘렀다)')

m.grad = None
(n_detached * 5).backward() if n_detached.requires_grad else print(
    'detach 한 값으로는 backward 자체가 안 됩니다 (끊겨 있으니까)')
print('m.grad =', m.grad, ' ← None. 아무것도 안 흘렀습니다.')

print('\n  ※ Actor-Critic 에서 advantage.detach() 를 빠뜨리면')
print('    "배우를 고치려던 신호가 평론가까지 흘러가" 학습이 망가집니다.')
print('    no_grad = 블록 전체를 끔 / detach = 특정 값 하나만 끊음')

print()
print('=' * 55)
print('6. 직접 경사하강법 한 번 돌려 보기 (옵티마이저 없이)')
print('=' * 55)

# y = (x-4)² 의 최솟값을 찾아봅니다. 답은 x=4 입니다.
x = torch.tensor(0.0, requires_grad=True)
lr = 0.1                        # 한 번에 얼마나 움직일지

print('  목표: y = (x-4)² 이 가장 작아지는 x 찾기 (정답 x=4)')
for step in range(15):
    y = (x - 4) ** 2            # ① 계산
    x.grad = None               # ② 지난 기울기 지우기
    y.backward()                # ③ 기울기 구하기
    with torch.no_grad():       # ④ 값을 직접 고칠 땐 미분 추적을 끈다
        x -= lr * x.grad        #    기울기 반대 방향으로 살짝 이동
    if step % 3 == 0:
        print(f'    step {step:2d}   x = {x.item():6.3f}   y = {y.item():7.4f}')

print(f'    최종      x = {x.item():.3f}  → 4에 가까워졌습니다')
print('\n  이 5단계가 모든 딥러닝 학습 루프의 전부입니다.')
print('  다음 파일(03, 04)에서는 이걸 신경망과 옵티마이저로 바꿔 씁니다.')
