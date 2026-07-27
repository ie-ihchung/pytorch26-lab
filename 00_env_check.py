# 설치 확인 — 아래가 에러 없이 실행되면 준비 완료
import torch
import gymnasium as gym
import numpy as np

print("PyTorch:", torch.__version__)
print("Gymnasium:", gym.__version__)
print("NumPy:", np.__version__)

# NumPy와 PyTorch가 서로 맞물리는지 확인합니다.
# 강화학습은 매 스텝 관측(NumPy)을 텐서로 바꾸기 때문에 이 줄이 실패하면
# 실습이 첫 줄부터 막힙니다. 실패하면: pip install "numpy<2"
print("NumPy 연동:", torch.from_numpy(np.array([1.0, 2.0], dtype=np.float32)))

x = torch.rand(5, 3)
print(x)                              # 랜덤 텐서가 출력되면 성공

# GPU(CUDA) 사용 가능 여부 확인 — 없어도 이번 과정은 CPU로 충분
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    # GPU가 달린 PC인데 cpu로 나온다면 CPU 전용 torch가 설치된 것입니다.
    # 환경설정 '필수 라이브러리 설치'의 --index-url 명령으로 다시 설치하세요.
    print("(GPU 없이 CPU로 진행합니다 — 이번 과정은 CPU로 충분합니다)")

# 3일 동안 실제로 쓰는 환경을 미리 전부 만들어 봅니다.
# LunarLander는 Box2D가 있어야 하므로 여기서 걸러집니다.
for env_id in ["CartPole-v1", "Pendulum-v1", "FrozenLake-v1",
               "CliffWalking-v1", "Taxi-v4", "MountainCar-v0", "LunarLander-v3"]:
    try:
        env = gym.make(env_id)
        env.reset(seed=0)
        env.close()
        print(f"  [OK] {env_id}")
    except Exception as e:
        print(f"  [실패] {env_id} — {type(e).__name__}: {e}")
