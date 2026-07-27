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

import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """경험 재현 버퍼 — DQN, DDPG, SAC 3일 내내 재사용합니다"""
    def __init__(self, capacity=100_000, action_dtype=torch.int64):
        # action_dtype: 오늘 DQN은 행동이 "몇 번 행동"인 정수라 int64입니다.
        # 3일차 DDPG·SAC는 행동이 연속값(실수 벡터)이므로 float32로 바꿔 씁니다
        #   buffer = ReplayBuffer(100_000, action_dtype=torch.float32)
        # int64로 두면 실수 행동이 정수로 잘려 학습이 통째로 망가집니다.
        self.buffer = deque(maxlen=capacity)
        self.action_dtype = action_dtype

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = zip(*batch)
        # np.array로 한 번 묶고 텐서로 바꿉니다.
        # 배열 리스트를 텐서로 바로 만들면 파이토치가 하나씩 복사해 매우 느립니다.
        return (torch.as_tensor(np.array(s), dtype=torch.float32),
                torch.as_tensor(np.array(a), dtype=self.action_dtype),
                torch.as_tensor(np.array(r), dtype=torch.float32),
                torch.as_tensor(np.array(s_next), dtype=torch.float32),
                torch.as_tensor(np.array(done), dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)
