/**
 * 교시별 "복사해서 바로 돌아가는" 자립 실행본을 만들어 사이트에 넣는다.
 *
 * 왜 필요한가.
 * 교시별 코드는 앞 교시의 변수·클래스를 이어 쓴다(1일차 3~5, 2일차 4·7, 3일차 2·6).
 * 그래서 한 교시 코드만 복사하면 NameError 가 난다.
 * 초보 수강생은 코드를 고쳐 맞출 수 없으므로, **그 교시까지의 코드를 합친 완성본**을
 * 각 교시에 붙여 준다. 복사 → 붙여넣기 → 실행이면 끝난다.
 *
 * 3일차는 2일차 1교시(ReplayBuffer)도 함께 넣는다 — 3일차만 열어도 돌아가게.
 *
 *   node scripts/gen_standalone.mjs          # 사이트 dayN.js 에 반영
 *   node scripts/gen_standalone.mjs --dry    # 미리보기
 */
import fs from 'node:fs'
import path from 'node:path'

const SITE = '/Volumes/aebon - 데이터/dreamit-web/01-edu-sites/pytorch26'
const DRY = process.argv.includes('--dry')

// 대상 지정: --only 1:5,6,7  → 1일차 5·6·7교시만
// 지정이 없으면 전 교시. 수업 진행에 맞춰 나눠 넣기 위한 장치.
const onlyArg = process.argv.find((a) => a.startsWith('--only='))
const ONLY = onlyArg
  ? Object.fromEntries(onlyArg.slice(7).split(';').map((part) => {
      const [d, slots] = part.split(':')
      return [Number(d), slots.split(',').map(Number)]
    }))
  : null

const { days } = await import(path.join(SITE, 'src/data/curriculum.js'))

// 3일차가 이어 쓰는 앞 일차 코드
const CARRY = { 3: [{ day: 2, slot: 1 }] }

const codeOf = (dayId, slot) => {
  const d = days.find((x) => x.id === dayId)
  return d?.sessions.find((s) => s.slot === slot)?.code
}

function buildStandalone(day, session) {
  const parts = []

  parts.push(`# ============================================================
# ${day.id}일차 ${session.slot}교시 — ${session.title}
# 복사해서 그대로 실행하면 됩니다. 고칠 것 없습니다.
# ------------------------------------------------------------
# 이 교시 코드는 앞 교시의 변수·클래스를 이어 씁니다.
# 그래서 이 블록에는 **여기까지 필요한 코드가 전부** 들어 있습니다.
# (수업용 코드만 따로 복사하면 NameError 가 납니다 — 그건 정상입니다.)
# ============================================================`)

  // 앞 일차에서 이어받는 것 (3일차 → 2일차 ReplayBuffer)
  for (const c of CARRY[day.id] ?? []) {
    const code = codeOf(c.day, c.slot)
    if (!code) continue
    parts.push(`\n# ── ${c.day}일차 ${c.slot}교시에서 이어받음 ──────────────────`)
    parts.push(code.source.trimEnd())
  }

  // 같은 일차의 1교시부터 이번 교시까지
  for (const s of day.sessions) {
    if (s.slot > session.slot) break
    if (!s.code) continue
    const tag = s.slot === session.slot ? '오늘 이 교시' : `${s.slot}교시에서 이어받음`
    parts.push(`\n# ── ${tag} — ${s.title} ──`)
    parts.push(s.code.source.trimEnd())
  }

  return parts.join('\n')
}

// ── 사이트 데이터에 삽입 ────────────────────────────────────
// 소스 문자열은 JS 템플릿 리터럴 안에 들어가므로 백틱·달러중괄호를 이스케이프한다.
const BT = String.fromCharCode(96)   // 백틱 — 리터럴로 쓰면 이 파일이 깨진다
const esc = (s) => s.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${')

let total = 0
for (const day of days) {
  const file = path.join(SITE, `src/data/day${day.id}.js`)
  let src = fs.readFileSync(file, 'utf8')

  // 뒤 교시부터 처리해야 앞쪽 인덱스가 밀리지 않는다
  let targets = [...day.sessions].filter((s) => s.code).reverse()
  if (ONLY) {
    const allow = ONLY[day.id]
    if (!allow) { console.log(`  ${day.id}일차 — 대상 아님, 건너뜀`); continue }
    targets = targets.filter((s) => allow.includes(s.slot))
  }

  for (const session of targets) {
    const slotIdx = src.indexOf(`slot: ${session.slot},`)
    if (slotIdx < 0) { console.log(`  slot ${session.slot} 못 찾음`); continue }

    // 이미 extraCode 가 있으면 건너뛴다 (4교시는 앞서 수동으로 넣었다)
    const nextSlotIdx = src.indexOf('\n      slot: ', slotIdx + 1)
    const region = src.slice(slotIdx, nextSlotIdx < 0 ? undefined : nextSlotIdx)
    if (region.includes('extraCode:')) {
      console.log(`  ${day.id}일차 ${session.slot}교시 — 이미 있음, 건너뜀`)
      continue
    }

    // code: { ... } 블록이 끝나는 위치 찾기
    const fnIdx = src.indexOf(`filename: '${session.code.filename}'`, slotIdx)
    if (fnIdx < 0) { console.log(`  ${session.slot}교시 filename 못 찾음`); continue }
    const closeIdx = src.indexOf('\n      },\n', fnIdx)
    if (closeIdx < 0) { console.log(`  ${session.slot}교시 code 끝 못 찾음`); continue }

    const standalone = buildStandalone(day, session)
    const DESC = [
      '앞 교시 코드까지 모두 포함된 ==완성본==입니다.',
      '복사 → 붙여넣기 → 실행. **고칠 것이 없습니다.**',
      '위 수업용 코드만 따로 복사하면 앞 교시 변수를 못 찾아 오류가 납니다 — 그건 정상입니다.',
    ].join('\n')

    const block = [
      '      extraCode: [',
      '        {',
      "          title: '이 교시 전체 코드 (복사해서 바로 실행)',",
      '          desc: ' + BT + DESC + BT + ',',
      "          filename: 'day" + day.id + '_' + String(session.slot).padStart(2, '0') + "_전체.py',",
      '          source: ' + BT + esc(standalone) + BT + ',',
      '        },',
      '      ],',
      '',
    ].join('\n')

    const insertAt = closeIdx + '\n      },\n'.length
    src = src.slice(0, insertAt) + block + src.slice(insertAt)
    total++
  }

  if (!DRY) fs.writeFileSync(file, src, 'utf8')
  console.log(`  ${day.id}일차 처리 완료`)
}

console.log(`\n${DRY ? '(미리보기) ' : ''}자립 실행본 ${total}개 삽입`)
