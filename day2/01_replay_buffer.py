# ==========================================================
# 2일차 1교시 — DQN 소개
# 2026-07-28 (화) 09:30 ~ 10:30 · Value-based & Policy-based Methods
# 원본 파일명: replay_buffer.py
# 사이트: https://pytorch26.dreamitbiz.com/#/day/2#s1
# ==========================================================
# [학습목표]
#  - 테이블 방식의 한계와 함수 근사의 필요성을 이해한다
#  - DQN의 핵심 요소인 경험 재현(Replay Buffer)과 타깃 네트워크를 설명할 수 있다
#
# ※ 이 파일은 같은 일차 앞 교시의 변수·클래스를 이어 씁니다.
#   단독 실행 시 NameError가 나면 정상입니다 — day2_전체.ipynb를 위에서부터 실행하세요.

import random                            # 여러 개 중에서 아무거나 뽑을 때 쓰는 도구
from collections import deque             # 앞뒤로 넣고 빼기 쉬운 '줄서기 상자'
import numpy as np                        # 숫자 계산 도구 (파이썬의 계산기)
import torch                              # 파이토치 — 신경망을 다루는 도구


class ReplayBuffer:
    """
    경험 재현 버퍼 — 게임하면서 겪은 일을 적어 두는 '일기장'입니다.

    왜 필요할까요?
      방금 겪은 일만 보고 배우면 비슷한 것만 연달아 보게 됩니다.
      (왼쪽으로 갔다 -> 또 왼쪽 -> 또 왼쪽 ...)
      그러면 신경망이 방금 본 것에만 맞추고 예전 것을 잊어버립니다.
      그래서 일기장에 잔뜩 적어 두고, 배울 때는 여기저기서 섞어서 꺼냅니다.

    이 상자는 3일 내내 씁니다 — 오늘 DQN, 내일 DDPG와 SAC.
    """

    def __init__(self, capacity=100_000, action_dtype=torch.int64):
        # capacity = 일기장에 몇 줄까지 적어 둘지 (10만 줄)
        #            넘치면 가장 오래된 것부터 자동으로 지워집니다.
        #
        # action_dtype = 행동을 어떤 숫자로 적을지
        #   오늘 DQN  : 행동이 "0번? 1번?" 이라서 정수(int64)
        #   내일 DDPG : 행동이 "힘을 1.37만큼" 이라서 실수(float32)
        #   -> 내일은 ReplayBuffer(100_000, action_dtype=torch.float32) 로 씁니다.
        #      정수로 두면 1.37 이 1 로 잘려서 학습이 통째로 망가집니다.
        self.buffer = deque(maxlen=capacity)    # 실제 일기장 (꽉 차면 앞쪽부터 밀려남)
        self.action_dtype = action_dtype        # 행동을 적을 숫자 종류를 기억해 둔다

    def push(self, s, a, r, s_next, done):
        # 일기 한 줄 적기:
        #   "이 상황(s)에서 이 행동(a)을 했더니 점수(r)를 받고
        #    저 상황(s_next)이 됐다. 그리고 판이 끝났나?(done)"
        self.buffer.append((s, a, r, s_next, done))   # 다섯 개를 한 묶음으로 저장

    def sample(self, batch_size):
        # 일기장에서 batch_size 줄을 무작위로 뽑아 온다 (= 섞어서 꺼내기)
        batch = random.sample(self.buffer, batch_size)   # 예: 아무 데서나 64줄

        # 뽑아온 것은 (상황, 행동, 점수, 다음상황, 끝났나) 묶음들의 목록입니다.
        # zip(*batch) 는 이걸 세로로 갈라 줍니다 —
        #   상황은 상황끼리, 행동은 행동끼리 따로 모아 줍니다.
        s, a, r, s_next, done = zip(*batch)

        # 파이썬 목록 -> 넘파이 배열 -> 파이토치 텐서 순서로 바꿉니다.
        # 왜 np.array 를 한 번 거칠까요?
        #   배열들의 '목록'을 텐서로 바로 만들면 파이토치가 하나씩 복사하느라
        #   아주 느려집니다. 넘파이로 먼저 한 덩어리를 만들면 훨씬 빠릅니다.
        return (
            torch.as_tensor(np.array(s), dtype=torch.float32),       # 상황들
            torch.as_tensor(np.array(a), dtype=self.action_dtype),   # 행동들
            torch.as_tensor(np.array(r), dtype=torch.float32),       # 점수들
            torch.as_tensor(np.array(s_next), dtype=torch.float32),  # 다음 상황들
            torch.as_tensor(np.array(done), dtype=torch.float32),    # 끝났나 (1이면 끝)
        )

    def __len__(self):
        # len(buffer) 라고 쓰면 이 함수가 불립니다 — 지금 몇 줄 적혀 있는지 알려 줍니다
        return len(self.buffer)

# ============================================================
# 잘 만들어졌는지 확인 (이 부분이 있어야 실행했을 때 결과가 보입니다)
# ============================================================
print('일기장이 잘 만들어졌는지 확인합니다.')
print()

buf = ReplayBuffer(capacity=1000)              # 1000줄짜리 일기장 하나 만들기

for i in range(200):                            # 가짜 경험 200줄 적어 보기
    s = np.random.randn(4).astype(np.float32)   # 상황 (숫자 4개)
    a = np.random.randint(2)                    # 행동 (0 또는 1)
    r = 1.0                                     # 점수
    s2 = np.random.randn(4).astype(np.float32)  # 다음 상황
    buf.push(s, a, r, s2, 0.0)                  # 일기장에 한 줄 적기

print(f'  일기장에 적힌 줄 수 : {len(buf)}')
print()

s, a, r, s2, done = buf.sample(32)              # 32줄 무작위로 꺼내기
print('  32줄 꺼냈을 때 모양')
print(f'    상황 s      {tuple(s.shape)}   {s.dtype}')
print(f'    행동 a      {tuple(a.shape)}      {a.dtype}   <- 정수여야 gather 가 됩니다')
print(f'    점수 r      {tuple(r.shape)}      {r.dtype}')
print(f'    다음상황 s2 {tuple(s2.shape)}   {s2.dtype}')
print(f'    끝났나 done {tuple(done.shape)}      {done.dtype}')
print()
print('  -> 모양이 위와 같이 나오면 정상입니다. 4교시에서 이 상자를 그대로 씁니다.')
