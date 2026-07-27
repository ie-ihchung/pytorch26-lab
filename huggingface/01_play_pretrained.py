# ============================================================
# [허깅페이스 ①] 남이 학습시킨 강화학습 모델을 받아서 바로 돌려 보기
# ------------------------------------------------------------
# 지금까지는 우리가 직접 학습시켰습니다. 몇 분씩 기다려야 했죠.
# 허깅페이스(Hugging Face)에는 ==이미 학습이 끝난 모델==이 올라와 있습니다.
# 받아서 바로 실행하면 학습 시간 0초로 만점짜리 플레이를 볼 수 있습니다.
#
# ★ 코랩에서 하세요 ★
#   설치 때문입니다. 아래 설명을 꼭 읽어 주세요.
#
# ┌──────────────────────────────────────────────────────────┐
# │ ⚠ 수업용 rl 환경(로컬)에는 절대 설치하지 마세요           │
# │                                                          │
# │   stable-baselines3 를 깔면 gymnasium 이 1.3 → 1.0 으로   │
# │   내려갑니다. 그러면 1일차에 쓴 CliffWalking-v1 과        │
# │   Taxi-v4 가 사라져서 1일차 코드가 전부 깨집니다.         │
# │   (강사 맥북에서 실제로 확인했습니다)                     │
# │                                                          │
# │   → 코랩에서 하거나, 별도 환경을 따로 만드세요.           │
# └──────────────────────────────────────────────────────────┘
#
# 코랩 첫 칸에 이것부터 실행:
#   !pip install -q stable-baselines3 huggingface_sb3
#
# 실행 시간: 설치 1분 + 실행 10초
# ============================================================
import numpy as np
import gymnasium as gym
from huggingface_sb3 import load_from_hub      # 허깅페이스에서 파일 받아오는 도구
from stable_baselines3 import DQN              # 남이 쓴 DQN 구현체

print('=' * 58)
print('1. 모델 내려받기')
print('=' * 58)

env = gym.make('CartPole-v1')

# repo_id = 허깅페이스에 올라간 주소, filename = 그 안의 파일 이름
checkpoint = load_from_hub(
    repo_id='sb3/dqn-CartPole-v1',
    filename='dqn-CartPole-v1.zip',
)
print('받은 파일:', checkpoint)

print()
print('=' * 58)
print('2. 불러오기 — custom_objects 가 필요한 이유')
print('=' * 58)

# 이 모델은 옛날 gym 으로 저장된 것이라 그대로 열면 오류가 납니다.
#   ModuleNotFoundError: No module named 'gym'
# 그래서 "그 부분은 지금 것으로 바꿔서 읽어라" 고 알려 줍니다.
custom_objects = {
    'learning_rate': 0.0,                       # 더 학습 안 할 거니 0
    'lr_schedule': lambda _: 0.0,               # 학습률 스케줄도 안 씀
    'exploration_schedule': lambda _: 0.0,      # 탐험도 안 함 (실력 그대로 보기)
    'observation_space': env.observation_space,  # 옛 gym 것 → 지금 gymnasium 것으로
    'action_space': env.action_space,
}

model = DQN.load(checkpoint, custom_objects=custom_objects)
print('불러오기 완료')
print('  ※ "SB3 < 2.4.0 으로 저장됨" 경고가 떠도 정상입니다. 그냥 돌아갑니다.')

print()
print('=' * 58)
print('3. 플레이 시켜 보기')
print('=' * 58)

returns = []
for ep in range(10):
    obs, _ = env.reset(seed=ep)
    total, done = 0.0, False
    while not done:
        # deterministic=True → 가장 좋다고 생각하는 행동만. 실력 측정용.
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(int(action))
        total += reward
        done = term or trunc
    returns.append(total)
    print(f'  {ep+1:2d}번째 판  점수 {total:5.0f}')

print(f'\n  평균 {np.mean(returns):.1f}   최고 {max(returns):.0f}')
print('  → CartPole-v1 의 만점은 500 입니다.')
print('  ※ 강사 맥북 실측: 10판 전부 500점.')
print('  ※ 우리가 2일차에 직접 만든 DQN 은 250판에 115점이었습니다.')
print('    같은 알고리즘이라도 오래 학습하면 이만큼 갑니다.')

print()
print('=' * 58)
print('4. 무작위로 하면 얼마나 나오나 (비교용)')
print('=' * 58)

rnd = []
for ep in range(10):
    obs, _ = env.reset(seed=100 + ep)
    total, done = 0.0, False
    while not done:
        obs, reward, term, trunc, _ = env.step(env.action_space.sample())
        total += reward
        done = term or trunc
    rnd.append(total)

print(f'  무작위 평균 {np.mean(rnd):.1f}')
print(f'  학습된 모델 {np.mean(returns):.1f}')
print('  → 이 차이가 "배웠다"는 뜻입니다.')
print('  ※ 강사 맥북 실측: 무작위 22.8점 / 학습된 모델 500점')

env.close()

# ============================================================
# 더 해보기
#   1) deterministic=False 로 바꿔 보세요 — 확률대로 뽑습니다. 점수가 조금 떨어집니다.
#   2) 다른 모델도 있습니다:
#        sb3/ppo-CartPole-v1        (PPO — 심화학습에서 다룹니다)
#        sb3/sac-Pendulum-v1        (SAC — 3일차 내용, 연속 행동)
#        sb3/td3-Pendulum-v1        (TD3 — DDPG 의 개선판)
#      강사 맥북 실측(5판 평균): sac-Pendulum-v1 -141.7, td3-Pendulum-v1 -145.6
#      (Pendulum 은 점수가 음수입니다. 0에 가까울수록 잘한 것.
#       무작위로 하면 -1180 이니 둘 다 아주 잘하는 것입니다)
#   3) 3일차를 듣고 나서 sac-Pendulum-v1 을 받아
#      우리가 만든 SAC 와 비교해 보세요.
# ============================================================
