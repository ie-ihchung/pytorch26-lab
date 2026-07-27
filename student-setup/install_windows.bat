@echo off
chcp 65001 > nul
REM ============================================================
REM  PyTorch로 배우는 강화학습 (2026-07-27~29)
REM  수강생 실습환경 설치 — Windows + NVIDIA GPU
REM  아나콘다(Anaconda Prompt)에서 실행하세요.
REM ============================================================

echo.
echo [1/5] 과정 전용 가상환경 rl 생성 (Python 3.10)
call conda create -n rl python=3.10 -y
if errorlevel 1 goto :err

echo.
echo [2/5] rl 환경 활성화
call conda activate rl
if errorlevel 1 goto :err

echo.
echo [3/5] PyTorch 설치 (CUDA 12.6 - RTX 2070용 GPU 빌드)
echo       ※ PyPI 기본 설치는 CPU 전용이라 반드시 index-url을 지정합니다.
pip install torch --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 goto :err

echo.
echo [4/5] 강화학습 환경 + 부속 라이브러리
pip install "gymnasium[box2d]" numpy matplotlib
if errorlevel 1 goto :err

echo.
echo [5/5] 설치 확인
python env_check.py

echo.
echo ============================================================
echo  설치가 끝났습니다.
echo  위 결과에 CUDA available: True 가 보이면 GPU까지 정상입니다.
echo ============================================================
pause
exit /b 0

:err
echo.
echo ------------------------------------------------------------
echo  설치 중 오류가 발생했습니다.
echo  Anaconda Prompt에서 실행했는지 확인하고,
echo  해결이 안 되면 위 메시지를 그대로 캡처해 강사에게 보내주세요.
echo ------------------------------------------------------------
pause
exit /b 1
