# ============================================================
# [파이토치 기초 ①] 텐서(Tensor) — 파이토치의 유일한 재료
# ------------------------------------------------------------
# 학습목표
#   · 텐서가 무엇인지 알고, 리스트/넘파이와 어떻게 다른지 설명할 수 있다
#   · 텐서를 만들고, 모양(shape)과 자료형(dtype)을 확인할 수 있다
#   · reshape / unsqueeze / squeeze 로 모양을 바꿀 수 있다
#   · 자주 나는 오류 두 가지(자료형 불일치, 모양 불일치)를 스스로 고칠 수 있다
#
# 실행 방법
#   코랩 새 노트북에 통째로 붙여넣고 Shift + Enter
#   설치할 것 없습니다. 파이토치는 코랩에 이미 들어 있습니다.
#
# 걸리는 시간: 1초
# ============================================================

import torch                    # 파이토치를 불러온다. 관례상 이름을 바꾸지 않고 그대로 쓴다.
import numpy as np              # 넘파이. 파이썬에서 숫자 계산을 빠르게 해주는 도구.

print('파이토치 버전:', torch.__version__)     # 지금 쓰고 있는 파이토치 버전을 찍어 본다
print('GPU 쓸 수 있나:', torch.cuda.is_available())   # True면 GPU, False면 CPU
print('  ※ False 여도 괜찮습니다. 이 과정 코드는 전부 CPU로 충분합니다.')


print()                         # 빈 줄 하나 (보기 좋으라고)
print('=' * 58)                 # '=' 를 58번 반복해 구분선을 그린다
print('1. 텐서 만들기 — 리스트를 상자에 담는다')
print('=' * 58)

a = torch.tensor([1.0, 2.0, 3.0])   # 파이썬 리스트를 텐서로 바꾼다
print('a          =', a)             # tensor([1., 2., 3.]) 라고 나온다
print('a의 모양   =', a.shape)       # shape = 몇 개짜리인지. torch.Size([3])
print('a의 자료형 =', a.dtype)       # dtype = 어떤 숫자인지. torch.float32

b = torch.tensor([1, 2, 3])          # 소수점을 안 찍으면?
print('b의 자료형 =', b.dtype)       # torch.int64 (정수)가 된다
print('  ※ 신경망은 float32 를 씁니다. 정수를 넣으면 오류가 납니다.')
print('    소수점을 찍거나 dtype=torch.float32 를 적어 주세요.')


print()
print('=' * 58)
print('2. 자주 쓰는 만들기 함수 네 가지')
print('=' * 58)

zeros = torch.zeros(2, 3)       # 0으로 채운 2줄 3칸짜리 표
print('torch.zeros(2, 3)  → 0으로 채우기')
print(zeros)

ones = torch.ones(2, 3)         # 1로 채운 2줄 3칸짜리 표
print('\ntorch.ones(2, 3)   → 1로 채우기')
print(ones)

randn = torch.randn(2, 3)       # 무작위 숫자로 채우기 (평균 0인 정규분포)
print('\ntorch.randn(2, 3)  → 무작위 (신경망의 처음 값이 이렇게 정해집니다)')
print(randn)

ar = torch.arange(5)            # 0, 1, 2, 3, 4
print('\ntorch.arange(5)    →', ar)


print()
print('=' * 58)
print('3. 모양(shape) — 오류의 절반이 여기서 납니다')
print('=' * 58)

x = torch.arange(12.0)          # 0.0 부터 11.0 까지 12개를 한 줄로
print('처음        모양', x.shape)
print(x)

y = x.reshape(3, 4)             # 3줄 4칸짜리 표로 바꾼다 (총 개수 12는 그대로)
print('\nreshape(3,4) 모양', y.shape)
print(y)
print('  ※ 총 개수가 맞아야 합니다. 12개를 (3,4)나 (2,6)으로는 되지만 (5,3)으로는 안 됩니다.')

print('\n─ unsqueeze / squeeze — 차원을 끼워 넣고 빼기 ─')
v = torch.tensor([1.0, 2.0, 3.0])       # 모양 (3,)  = 그냥 숫자 3개
print('  원래         ', v.shape, '  숫자 3개가 한 줄로')
print('  unsqueeze(0) ', v.unsqueeze(0).shape, ' 가로로 눕힌 1줄짜리 표')
print('  unsqueeze(1) ', v.unsqueeze(1).shape, ' 세로로 세운 3줄짜리 표')
print('  다시 squeeze ', v.unsqueeze(1).squeeze().shape, '  1인 차원을 없애 원래대로')
print('  ※ 강화학습 코드에 unsqueeze(1) 이 계속 나옵니다.')
print('    "여러 개를 한 번에" 처리하는 모양을 맞추기 위해서입니다.')


print()
print('=' * 58)
print('4. 계산 — 넘파이와 거의 같습니다')
print('=' * 58)

p = torch.tensor([1.0, 2.0, 3.0])
q = torch.tensor([10.0, 20.0, 30.0])

print('p =', p)
print('q =', q)
print('p + q      =', p + q)        # 같은 자리끼리 더한다
print('p * q      =', p * q)        # 같은 자리끼리 곱한다 (행렬 곱이 아닙니다!)
print('p @ q      =', p @ q)        # 내적: 1*10 + 2*20 + 3*30 = 140
print('p.sum()    =', p.sum())      # 다 더하기
print('p.mean()   =', p.mean())     # 평균
print('p.max()    =', p.max())      # 가장 큰 값
print('p.argmax() =', p.argmax())   # 가장 큰 값이 몇 번째인지 (0부터 셈)
print('  ※ argmax 는 강화학습에서 "가장 좋은 행동 고르기"에 씁니다.')


print()
print('=' * 58)
print('5. 넘파이와 오가기 — 환경과 신경망 사이의 통역')
print('=' * 58)

n = np.array([1.0, 2.0, 3.0], dtype=np.float32)   # 넘파이 배열을 하나 만든다
t = torch.from_numpy(n)                            # 넘파이 → 텐서
print('넘파이 → 텐서 :', t)
print('텐서 → 넘파이 :', t.numpy())                # 텐서 → 넘파이
print('  ※ 게임 환경(gymnasium)은 넘파이를 내놓고 신경망은 텐서를 받습니다.')
print('    그래서 이 변환이 매 걸음 들어갑니다.')


print()
print('=' * 58)
print('6. 오류를 일부러 내 보기 — 미리 겪어 두면 안 무섭습니다')
print('=' * 58)

try:                                        # try = "해보고 오류가 나면 아래 except 로"
    torch.tensor([1, 2, 3]) @ torch.tensor([1.0, 2.0, 3.0])   # 정수 @ 실수
except RuntimeError as e:                   # RuntimeError 라는 오류가 나면 여기로 온다
    print('[오류 1] 정수와 실수를 섞었을 때')
    print('  메시지:', str(e)[:64])
    print('  고치기: torch.tensor([1,2,3], dtype=torch.float32)')

try:
    torch.zeros(2, 3) + torch.zeros(4, 5)   # 모양이 안 맞는 것끼리 더하기
except RuntimeError as e:
    print('\n[오류 2] 모양이 안 맞을 때')
    print('  메시지:', str(e)[:64])
    print('  고치기: print(x.shape) 로 찍어 보고 reshape / unsqueeze 로 맞춘다')

print('\n※ 막히면 무조건 print(x.shape) 부터. 절반은 여기서 해결됩니다.')
