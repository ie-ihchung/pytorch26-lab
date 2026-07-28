/**
 * experiments/*.py 를 사이트의 해당 교시 extraCode 에 붙인다.
 *
 * 전체본(gen_standalone.mjs)과 역할이 다르다.
 *   전체본  = 그 교시 수업 코드를 그대로 돌려 보는 것
 *   실험    = 값을 바꿔 가며 "왜 이렇게 하는지"를 눈으로 보는 것
 * 둘 다 있어야 개별 실습이 된다.
 *
 * 이미 같은 title 의 항목이 있으면 source 만 갱신한다(재실행 안전).
 *
 *   node scripts/gen_experiments.mjs
 *   node scripts/gen_experiments.mjs --dry
 */
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const LAB = path.join(HERE, '..')
const SITE = '/Volumes/aebon - 데이터/dreamit-web/01-edu-sites/pytorch26'
const DRY = process.argv.includes('--dry')

const BT = String.fromCharCode(96)
const esc = (s) => s.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${')

const EXPERIMENTS = [
  {
    day: 2, slot: 1, file: 'day2_01_why_replay.py',
    title: '실험 — 경험 재현이 왜 필요한가 (11초)',
    desc: `강화학습 코드가 아닙니다. ==곡선 맞추기 하나로== 버퍼가 왜 필요한지만 봅니다.

**순서대로 준 데이터**와 **섞어서 준 데이터**로 같은 신경망을 학습시킵니다.
강사 맥북 실측: 순서대로 **0.0062**, 섞어서 **0.0000** — 학습곡선 2개가 함께 나옵니다.`,
  },
  {
    day: 2, slot: 2, file: 'day2_02_max_bias.py',
    title: '실험 — Q값이 부풀려지는 것을 숫자로 (7초)',
    desc: `신경망도 강화학습도 없습니다. **주사위만 던집니다.**

진짜 가치가 전부 0인 행동들을 몇 번씩 해보고 max 를 취하면 어떻게 되는지 봅니다.
강사 맥북 실측: 행동 50개일 때 그냥 max는 **+0.50**, Double 방식은 **-0.008**.
==Double DQN 이 왜 필요한지가 이 표 하나로 끝납니다.==`,
  },
  {
    day: 2, slot: 4, file: 'day2_04_target_network.py',
    title: '실험 — 타깃 네트워크가 있고 없고 (27초)',
    desc: `같은 DQN 을 두 번 돌립니다. ==딱 한 줄만 다릅니다.==

\`nxt = tgt(bs2).max(...)\` 인지 \`nxt = q(bs2).max(...)\` 인지.
강사 맥북 실측: 있음 **115.6점**, 없음 **10.7점** (없는 쪽은 처음보다 오히려 나빠졌습니다).
학습곡선 2개를 겹쳐 그려 줍니다.`,
  },
  {
    day: 2, slot: 5, file: 'day2_05_baseline.py',
    title: '실험 — 베이스라인을 빼면 왜 안정되나 (72초)',
    desc: `REINFORCE 를 두 번 돌립니다. **바뀌는 건 한 줄뿐입니다.**

\`G = (G - G.mean()) / G.std()\` 를 넣느냐 마느냐.
강사 맥북 실측: 기울기 크기가 **738 → 22.6** 으로 줄었습니다.
점수 곡선과 기울기 곡선을 나란히 보여 줍니다.`,
  },
  {
    day: 2, slot: 6, file: 'day2_06_entropy.py',
    title: '실험 — 엔트로피 보너스를 0 / 0.01 / 0.1 로 (48초)',
    desc: `A2C 손실의 마지막 항 \`- c × 엔트로피\` 만 바꿔 세 번 돌립니다.

==결과가 교과서 설명대로 나오지 않습니다.== 그게 이 실험의 핵심입니다.
강사 맥북 실측: c=0 **106.7점**, c=0.01 **97.0점**, c=0.1 **77.5점**.
엔트로피는 0.605 / 0.583 / 0.580 으로 거의 같았습니다.
**하이퍼파라미터는 남이 쓴 값을 믿지 말고 직접 돌려 봐야 한다** — 이걸 배우는 실습입니다.`,
  },
  {
    day: 2, slot: 6, file: 'day2_06_detach.py',
    title: '실험 — detach() 를 빼면 무슨 일이 나는가 (90초)',
    desc: `같은 A2C 를 두 번 돌립니다. **다른 곳은 딱 한 줄입니다.**

\`advantage.detach()\` 인지 그냥 \`advantage\` 인지.

강사 맥북 실측(seed=0):
있음 **17.2점** / 없음 **9.5점** — ==점수가 절반 아래로 떨어집니다.==
평론가 손실은 **877 → 2411** 로 세 배 가까이 커집니다.

★ 그런데 **오류가 한 줄도 안 납니다.** 그냥 조용히 나빠집니다.
초보자가 "손실이 줄어드니 잘 되고 있네" 하고 넘어가기 딱 좋은 실수입니다.`,
  },
  {
    day: 3, slot: 1, file: 'day3_01_soft_update.py',
    title: '실험 — 과녁을 통째로 갈까, 조금씩 섞을까 (3초)',
    desc: `어제는 20판마다 과녁을 **통째로** 갈았고, 오늘은 매번 **0.5%씩** 섞습니다.
왜 바꾸는지를 숫자와 그림으로 봅니다. 강화학습 코드가 아니라 단순한 따라가기 문제입니다.

강사 맥북 실측: 하드 방식의 가장 큰 변화가 소프트의 **93배**였습니다.
==다만 소프트는 부드러운 대신 뒤처집니다. 공짜가 아닙니다.==`,
  },
  {
    day: 3, slot: 4, file: 'day3_04_rsample.py',
    title: '실험 — rsample 과 tanh 보정 (8초)',
    desc: `SAC 코드에서 **가장 많이 틀리는 두 곳**을 눈으로 봅니다.

**①** \`sample()\` 을 쓰면 배우가 아예 학습되지 않습니다 — 직접 확인합니다.
**②** tanh 보정을 빼면 확률이 틀어집니다 — ==적분해서 검산까지 합니다.==
강사 맥북 실측: 보정하면 0.4163, 안 하면 0.3805 (정답은 표본값 0.4158).`,
  },
  {
    day: 3, slot: 6, file: 'day3_06_twin_q.py',
    title: '실험 — 평론가를 두 명 두면 (3초)',
    desc: `신경망 없이 **주사위만** 던져 트윈 Q 의 효과를 확인합니다.

강사 맥북 실측(행동 10개, 잡음 1.0):
평론가 1명 **+1.54** / 두 명 중 낮은 쪽 **+0.66** / 두 명 평균 **+1.10**
==부풀림이 절반 이하로 줄지만 완전히 없어지지는 않습니다.==
왜 평균이 아니라 최솟값을 쓰는지가 이 표에 있습니다.`,
  },
]

let changed = 0
for (const exp of EXPERIMENTS) {
  const file = path.join(SITE, `src/data/day${exp.day}.js`)
  let src = fs.readFileSync(file, 'utf8')

  const source = fs.readFileSync(path.join(LAB, 'experiments', exp.file), 'utf8').trimEnd()

  const slotIdx = src.indexOf(`slot: ${exp.slot},`)
  if (slotIdx < 0) { console.log(`  ${exp.day}-${exp.slot}: slot 못 찾음`); continue }

  // 이 교시 구역만 잘라 본다 (다음 slot 전까지)
  const nextIdx = src.indexOf('\n      slot: ', slotIdx + 1)
  const end = nextIdx < 0 ? src.length : nextIdx

  if (src.slice(slotIdx, end).includes(exp.title)) {
    console.log(`  ${exp.day}일차 ${exp.slot}교시 — 이미 있음, 건너뜀`)
    continue
  }

  // extraCode 배열의 여는 줄 뒤에 새 항목을 끼워 넣는다
  const arrIdx = src.indexOf('      extraCode: [', slotIdx)
  if (arrIdx < 0 || arrIdx > end) {
    console.log(`  ${exp.day}일차 ${exp.slot}교시 — extraCode 배열 없음 (먼저 gen_standalone 실행)`)
    continue
  }
  const insertAt = arrIdx + '      extraCode: [\n'.length

  const block = [
    '        {',
    `          title: '${exp.title}',`,
    '          desc: ' + BT + esc(exp.desc) + BT + ',',
    `          filename: '${exp.file}',`,
    '          source: ' + BT + esc(source) + BT + ',',
    '        },',
    '',
  ].join('\n')

  src = src.slice(0, insertAt) + block + src.slice(insertAt)
  if (!DRY) fs.writeFileSync(file, src, 'utf8')
  changed++
  console.log(`  ${exp.day}일차 ${exp.slot}교시 ← ${exp.file}`)
}

console.log(`\n${DRY ? '(미리보기) ' : ''}실험 ${changed}개 삽입`)
