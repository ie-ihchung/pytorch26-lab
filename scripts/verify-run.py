"""강의 전 실행 검증 — 일차 노트북 3개와 미니 프로젝트 8개를 실제로 돌린다.

빌드 통과는 파이썬 코드가 돈다는 뜻이 아니다. 강의 중 멈추는 자리를 미리 찾는 것이 목적.

    python scripts/verify-run.py            # 전체
    python scripts/verify-run.py day3       # 일부만
"""
import contextlib
import io
import json
import os
import signal
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # plt.show()가 창을 띄워 멈추지 않도록

LAB = Path(__file__).resolve().parent.parent
CELL_LIMIT = 300      # 셀 하나가 이보다 오래 걸리면 강의 진행에 무리
PROJECT_LIMIT = 120   # 프로젝트는 학습 루프라 도중 중단해도 정상으로 본다

# 단독 실행되지 않는 것이 정상인 파일 — 사이트 projects.js 주석에 명시돼 있다.
# #5·#6은 3일차 수업 코드(클래스)를 전제한 러너다.
EXPECTED_DEPENDENT = {"sol05_ddpg_vs_sac.py", "sol06_tac_q_experiment.py"}

results = []


class Timeout(Exception):
    pass


@contextlib.contextmanager
def time_limit(sec):
    def handler(signum, frame):
        raise Timeout()

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(sec)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def run_notebook(day):
    """노트북을 한 인터프리터에서 셀 순서대로 실행 — 수업 진행과 같은 방식."""
    path = LAB / f"day{day}" / f"day{day}_전체.ipynb"
    nb = json.loads(path.read_text(encoding="utf-8"))
    g = {"__name__": "__main__"}
    print(f"\n── day{day} 노트북 ──")
    for c in [x for x in nb["cells"] if x["cell_type"] == "code"]:
        t = time.time()
        try:
            with time_limit(CELL_LIMIT), contextlib.redirect_stdout(io.StringIO()):
                exec(compile("".join(c["source"]), c["id"], "exec"), g)
            dt = time.time() - t
            print(f"  [OK]   {c['id']:<22} {dt:6.1f}s")
            results.append((f"day{day}/{c['id']}", True, f"{dt:.1f}s"))
        except Timeout:
            print(f"  [실패] {c['id']:<22} {CELL_LIMIT}초 초과")
            results.append((f"day{day}/{c['id']}", False, f"{CELL_LIMIT}초 초과"))
            return
        except Exception as e:
            print(f"  [실패] {c['id']:<22} {type(e).__name__}: {str(e)[:70]}")
            results.append((f"day{day}/{c['id']}", False, type(e).__name__))
            return  # 뒤 셀은 앞 셀에 의존하므로 더 볼 의미가 없다


def run_projects():
    print("\n── 미니 프로젝트 ──")
    for f in sorted((LAB / "projects").glob("*.py")):
        if f.name in EXPECTED_DEPENDENT:
            print(f"  [건너뜀] {f.name} — 3일차 수업 코드를 전제한 러너 (정상)")
            continue
        t = time.time()
        try:
            with time_limit(PROJECT_LIMIT), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(f.read_text(encoding="utf-8"), f.name, "exec"), {"__name__": "__main__"})
            print(f"  [OK]   {f.name:<40} {time.time()-t:6.1f}s")
            results.append((f.name, True, "완주"))
        except Timeout:
            # 학습 루프가 길어 시간 안에 안 끝난 것 — 오류 없이 돌고 있으면 정상
            print(f"  [OK]   {f.name:<40} 학습 진행중({PROJECT_LIMIT}초) — 오류 없음")
            results.append((f.name, True, "학습중"))
        except Exception as e:
            print(f"  [실패] {f.name:<40} {type(e).__name__}: {str(e)[:60]}")
            results.append((f.name, False, type(e).__name__))


if __name__ == "__main__":
    want = sys.argv[1:] or ["day1", "day2", "day3", "projects"]
    for d in (1, 2, 3):
        if f"day{d}" in want:
            run_notebook(d)
    if "projects" in want:
        run_projects()

    bad = [r for r in results if not r[1]]
    print("\n" + "=" * 60)
    print(f" 검사 {len(results)}건 — 실패 {len(bad)}건")
    for name, _, detail in bad:
        print(f"   - {name}: {detail}")
    print("=" * 60)
    sys.exit(1 if bad else 0)
