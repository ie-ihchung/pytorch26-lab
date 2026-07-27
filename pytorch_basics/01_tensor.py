# ============================================================
# [파이토치 기초 ①] 텐서 — 파이토치의 유일한 재료
# ------------------------------------------------------------
# 파이토치에서 다루는 것은 딱 하나, '텐서(tensor)'입니다.
# 텐서 = 숫자를 담는 상자. 엑셀 표를 떠올리시면 됩니다.
#   숫자 하나         → 0차원 (스칼라)
#   한 줄             → 1차원 (벡터)
#   표                → 2차원 (행렬)
#   표를 여러 장 쌓음 → 3차원 이상
#
# 실행: python 01_tensor.py    (1초)
# ============================================================
import torch                      # 파이토치를 불러온다. 관례상 이름 그대로 씁니다.
import numpy as np                # 넘파이 — 파이썬의 표준 숫자 라이브러리

print('=' * 55)
print('1. 텐서 만들기')
print('=' * 55)

a = torch.tensor([1.0, 2.0, 3.0])       # 파이썬 리스트 → 텐서
print('a          =', a)                 # tensor([1., 2., 3.])
print('a의 모양   =', a.shape)           # torch.Size([3])  → 숫자 3개짜리 한 줄
print('a의 자료형 =', a.dtype)           # torch.float32    → 소수점 있는 숫자

b = torch.tensor([1, 2, 3])              # 소수점을 안 찍으면 정수 텐서가 된다
print('b의 자료형 =', b.dtype)           # torch.int64
print('  ※ 신경망은 float32 를 씁니다. 정수를 넣으면 오류가 납니다.')

print()
print('=' * 55)
print('2. 자주 쓰는 만들기 함수')
print('=' * 55)

print('0으로 채우기      torch.zeros(2, 3) →\n', torch.zeros(2, 3))
print('1로 채우기        torch.ones(2, 3)  →\n', torch.ones(2, 3))
print('무작위(정규분포)  torch.randn(2, 3) →\n', torch.randn(2, 3))
print('0,1,2,...        torch.arange(5)   →', torch.arange(5))

print()
print('=' * 55)
print('3. 모양(shape) — 오류의 90%가 여기서 납니다')
print('=' * 55)

x = torch.arange(12.0)                   # 0부터 11까지 12개
print('처음          ', x.shape, x)

y = x.reshape(3, 4)                      # 3줄 4칸짜리 표로 바꾼다 (총 개수는 그대로 12)
print('reshape(3,4)  ', y.shape)
print(y)

print('\n unsqueeze / squeeze — 차원을 끼워 넣고 빼기')
v = torch.tensor([1.0, 2.0, 3.0])        # shape (3,)
print('  원래          ', v.shape)
print('  unsqueeze(0)  ', v.unsqueeze(0).shape, ' → 세로 1줄짜리 표 (1, 3)')
print('  unsqueeze(1)  ', v.unsqueeze(1).shape, ' → 가로 1칸짜리 표 (3, 1)')
print('  다시 squeeze  ', v.unsqueeze(1).squeeze().shape)
print('  ※ 강화학습 코드에서 unsqueeze(1) 이 자주 나옵니다.')
print('    신경망이 "여러 개를 한 번에" 처리하는 모양을 맞추기 위해서입니다.')

print()
print('=' * 55)
print('4. 계산 — 넘파이와 거의 같습니다')
print('=' * 55)

p = torch.tensor([1.0, 2.0, 3.0])
q = torch.tensor([10.0, 20.0, 30.0])
print('p + q      =', p + q)             # 같은 자리끼리 더한다
print('p * q      =', p * q)             # 같은 자리끼리 곱한다 (행렬곱 아님!)
print('p @ q      =', p @ q)             # 내적 = 1*10 + 2*20 + 3*30 = 140
print('p.sum()    =', p.sum())
print('p.mean()   =', p.mean())
print('p.max()    =', p.max())
print('p.argmax() =', p.argmax(), ' → 가장 큰 값이 몇 번째인지 (RL에서 행동 고를 때 씁니다)')

print()
print('=' * 55)
print('5. 넘파이와 오가기')
print('=' * 55)

n = np.array([1.0, 2.0, 3.0], dtype=np.float32)
t = torch.from_numpy(n)                  # 넘파이 → 텐서
print('넘파이 → 텐서 :', t)
print('텐서 → 넘파이 :', t.numpy())
print('  ※ gymnasium 환경은 넘파이를 내놓습니다. 그래서 이 변환이 매번 필요합니다.')

print()
print('=' * 55)
print('6. 실전에서 만나는 오류 재현해 보기')
print('=' * 55)

try:
    torch.tensor([1, 2, 3]) @ torch.tensor([1.0, 2.0, 3.0])   # 정수 @ 실수
except RuntimeError as e:
    print('정수와 실수를 섞으면:', str(e)[:70])
    print('  해결: torch.tensor([1,2,3], dtype=torch.float32)')

try:
    torch.zeros(2, 3) + torch.zeros(4, 5)                      # 모양이 안 맞음
except RuntimeError as e:
    print('\n모양이 안 맞으면:', str(e)[:70])
    print('  해결: .shape 를 찍어 보고 reshape / unsqueeze 로 맞춥니다')

print('\n※ 막히면 무조건 print(x.shape) 부터 찍어 보세요. 절반은 여기서 해결됩니다.')
