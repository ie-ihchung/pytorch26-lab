/**
 * 이미 넣어 둔 "전체 코드" 블록을 최신 수업 코드로 갱신한다.
 * 수업 코드에 주석을 더하면 전체본도 같이 바뀌어야 하는데,
 * gen_standalone 은 이미 있는 블록을 건너뛰므로 갱신용을 따로 둔다.
 *
 *   node scripts/refresh_standalone.mjs 1:5,6,7
 */
import fs from 'node:fs'
import path from 'node:path'

const SITE = '/Volumes/aebon - 데이터/dreamit-web/01-edu-sites/pytorch26'
const { days } = await import(path.join(SITE, 'src/data/curriculum.js'))
const CARRY = { 3: [{ day: 2, slot: 1 }] }
const BT = String.fromCharCode(96)
const esc = (s) => s.replace(/\\/g, '\\\\').replace(/`/g, '\\`').replace(/\$\{/g, '\\${')

const codeOf = (d, sl) => days.find((x) => x.id === d)?.sessions.find((s) => s.slot === sl)?.code

function build(day, session) {
  const parts = [`# ============================================================
# ${day.id}일차 ${session.slot}교시 — ${session.title}
# 복사해서 그대로 실행하면 됩니다. 고칠 것 없습니다.
# ------------------------------------------------------------
# 이 교시 코드는 앞 교시의 변수·클래스를 이어 씁니다.
# 그래서 이 블록에는 **여기까지 필요한 코드가 전부** 들어 있습니다.
# (수업용 코드만 따로 복사하면 NameError 가 납니다 — 그건 정상입니다.)
# ============================================================`]
  for (const c of CARRY[day.id] ?? []) {
    const code = codeOf(c.day, c.slot)
    if (code) { parts.push(`\n# ── ${c.day}일차 ${c.slot}교시에서 이어받음 ──`); parts.push(code.source.trimEnd()) }
  }
  for (const s of day.sessions) {
    if (s.slot > session.slot) break
    if (!s.code) continue
    parts.push(`\n# ── ${s.slot === session.slot ? '오늘 이 교시' : s.slot + '교시에서 이어받음'} — ${s.title} ──`)
    parts.push(s.code.source.trimEnd())
  }
  return parts.join('\n')
}

/** 이스케이프되지 않은 닫는 백틱 위치 */
function endOfLiteral(src, from) {
  for (let i = from; i < src.length; i++) {
    if (src[i] === BT && src[i - 1] !== '\\') return i
  }
  return -1
}

const [target] = process.argv.slice(2)
const [dayId, slotList] = target.split(':')
const slots = slotList.split(',').map(Number)
const day = days.find((d) => d.id === Number(dayId))
const file = path.join(SITE, `src/data/day${day.id}.js`)
let src = fs.readFileSync(file, 'utf8')

for (const slot of [...slots].sort((a, b) => b - a)) {
  const session = day.sessions.find((s) => s.slot === slot)
  const slotIdx = src.indexOf(`slot: ${slot},`)
  const exIdx = src.indexOf('extraCode: [', slotIdx)
  if (exIdx < 0) { console.log(`  ${slot}교시 extraCode 없음`); continue }
  const srcKey = src.indexOf('source: ' + BT, exIdx)
  const start = srcKey + ('source: ' + BT).length
  const end = endOfLiteral(src, start)
  src = src.slice(0, start) + esc(build(day, session)) + src.slice(end)
  console.log(`  ${day.id}일차 ${slot}교시 전체본 갱신`)
}

fs.writeFileSync(file, src, 'utf8')
