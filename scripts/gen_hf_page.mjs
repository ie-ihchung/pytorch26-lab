/**
 * huggingface/*.py 를 읽어 사이트의 src/data/huggingface.js 를 만든다.
 * (gen_pytorch_page.mjs 와 같은 구조 — .py 가 정본, 사이트는 사본)
 *
 *   node scripts/gen_hf_page.mjs
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const LAB = path.join(HERE, '..')
const SITE = '/Volumes/aebon - 데이터/dreamit-web/01-edu-sites/pytorch26'

const BT = String.fromCharCode(96)
const esc = (s) => s.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${')

const SECTIONS = [
  {
    file: '01_play_pretrained.py',
    title: '학습 0초로 만점 플레이 보기',
    minutes: 15,
    goal: '이미 학습이 끝난 모델을 받아서 바로 돌립니다.',
    why: `지금까지는 우리가 직접 학습시켰습니다. 몇 분씩 기다려야 했죠.

허깅페이스에는 ==이미 학습이 끝난 모델==이 올라와 있습니다.
받아서 실행하면 **학습 시간 0초**로 만점짜리 플레이를 볼 수 있습니다.

**강사 맥북 실측**: 10판 전부 500점(만점). 무작위로 하면 22.8점입니다.
우리가 2일차에 직접 만든 DQN은 250판 학습해서 115점이었습니다.
==같은 알고리즘이라도 충분히 학습하면 여기까지 갑니다.==`,
    tryIt: `\`deterministic=True\` 를 \`False\` 로 바꿔 보세요. 확률대로 행동하게 되어 점수가 조금 떨어집니다.
"실력을 재는 것"과 "탐험하며 행동하는 것"이 다르다는 걸 여기서도 확인할 수 있습니다.`,
  },
  {
    file: '02_compare_algorithms.py',
    title: '알고리즘을 나란히 비교하기 (3일차 예습)',
    minutes: 15,
    goal: '3일차에 배울 SAC·TD3를 미리 돌려 봅니다.',
    why: `Pendulum은 **막대를 흔들어 거꾸로 세우는** 문제입니다. 3일차에 쓰는 환경입니다.
행동이 왼쪽/오른쪽이 아니라 ==힘을 얼마나 줄지(연속값)== 입니다.

직접 학습시키면 알고리즘마다 몇 분씩 걸리는데, 받아 오면 30초면 비교가 끝납니다.

**강사 맥북 실측(5판)**: SAC **-141.7**, TD3 **-145.6**, 무작위 **-1180.4**
Pendulum은 점수가 항상 음수입니다. 0에 가까울수록 잘한 것입니다.`,
    tryIt: `\`episodes\` 를 20으로 늘려 보세요. 평균이 얼마나 흔들리나요?
TD3는 한 판에서 -1점, 다른 판에서 -245점이 나왔습니다.
==몇 판 돌려서 판단해야 하는지== 감을 잡는 실습입니다.`,
  },
]

const parts = SECTIONS.map((s) => {
  const src = fs.readFileSync(path.join(LAB, 'huggingface', s.file), 'utf8').trimEnd()
  return `  {
    id: '${s.file.replace('.py', '')}',
    title: ${BT}${esc(s.title)}${BT},
    minutes: ${s.minutes},
    goal: ${BT}${esc(s.goal)}${BT},
    why: ${BT}${esc(s.why)}${BT},
    tryIt: ${BT}${esc(s.tryIt)}${BT},
    code: {
      filename: '${s.file}',
      source: ${BT}${esc(src)}${BT},
    },
  },`
}).join('\n')

const out = `// ============================================================
// 허깅페이스 실습 — 자동 생성 파일
// ------------------------------------------------------------
// 정본은 pytorch26-lab 리포의 huggingface/*.py 이며 전부 실제로
// 실행해 검증했다. 고칠 때는 그쪽을 고치고 아래를 다시 돌린다.
//
//   node scripts/gen_hf_page.mjs
//
// 왜 코랩 전용인가 (2026-07-28 실측):
//   stable-baselines3 를 설치하면 gymnasium 이 1.3 → 1.0 으로 내려간다.
//   그러면 1일차에 쓴 CliffWalking-v1 · Taxi-v4 가 레지스트리에서 사라져
//   1일차 코드가 통째로 깨진다. 수업용 rl 환경에 절대 설치하면 안 된다.
// ============================================================

export default [
${parts}
]
`

fs.writeFileSync(path.join(SITE, 'src/data/huggingface.js'), out, 'utf8')
console.log(`src/data/huggingface.js 생성 — ${SECTIONS.length}개 절`)
