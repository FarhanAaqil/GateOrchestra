import { Card, ErrorNote, Page, PageHeader, SkeletonCards } from '../components/ui'
import { usageApi, type UsageCounts } from '../lib/account'
import { useApi } from '../lib/useApi'

function Meter({ label, used, limit, unit }: { label: string; used: number; limit: number; unit: string }) {
  const share = limit ? Math.min(1, used / limit) : 0
  return (
    <Card tilt={false}>
      <p className="text-sm text-muted">{label}</p>
      <p className="num mt-2 text-[28px] font-medium leading-none text-ink">{used.toLocaleString()}</p>
      <p className="mt-1 font-mono text-xs text-muted">of {limit.toLocaleString()} {unit} today</p>
      <div className="mt-3 h-1.5 rounded-full bg-forest/10" role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={limit} aria-valuenow={used}>
        <div className={`h-1.5 rounded-full ${share >= 0.9 ? 'bg-wine' : 'bg-forest'}`} style={{ width: `${share * 100}%` }} />
      </div>
    </Card>
  )
}

function Routes({ month }: { month: UsageCounts }) {
  const rows = [
    { label: 'Answered directly', value: month.direct, note: 'the quick check was confident, so no research agent ran' },
    { label: 'Researched with tools', value: month.escalated, note: 'escalated to the research agent' },
    { label: 'Small talk', value: month.small_talk, note: 'greetings and thanks' },
    { label: 'From the cache', value: month.cached, note: 'a repeat of your own earlier question' },
  ]
  const total = rows.reduce((sum, r) => sum + r.value, 0)
  return (
    <Card tilt={false}>
      <h2 className="text-lg font-semibold text-ink">How your answers were handled this month</h2>
      {total ? (
        <ul className="mt-3 space-y-2.5">
          {rows.map((r) => (
            <li key={r.label} className="text-sm">
              <div className="flex justify-between text-forest">
                <span>{r.label} <span className="text-xs text-muted">· {r.note}</span></span>
                <span className="font-mono text-xs">{r.value} · {Math.round((r.value / total) * 100)}%</span>
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-forest/10">
                <div className="h-1.5 rounded-full bg-forest" style={{ width: `${(r.value / total) * 100}%` }} />
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="mt-2 text-sm text-muted">No answers yet this month.</p>
      )}
    </Card>
  )
}

export default function Usage() {
  const usage = useApi(usageApi.usage)
  const u = usage.data
  return (
    <Page>
      <PageHeader kicker="Account" title="Your usage">
        What you have used today and this month. Days run on UTC, and limits reset at midnight UTC.
      </PageHeader>
      {usage.error ? <ErrorNote message={usage.error} /> : null}
      {usage.loading && !u ? <SkeletonCards count={3} height="h-24" /> : null}
      {u ? (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Card tilt={false}>
              <p className="text-sm text-muted">Plan</p>
              <p className="mt-2 text-xl font-semibold text-ink">{u.plan.label}</p>
              <p className="mt-1 font-mono text-xs text-muted">
                {u.paid_until ? `paid until ${new Date(u.paid_until * 1000).toLocaleDateString()}` : u.plan.price_usd_month ? '' : 'free'}
              </p>
            </Card>
            <Meter label="Messages" used={u.today.messages} limit={u.plan.messages_per_day} unit="messages" />
            <Meter label="Tokens on our keys" used={u.today.tokens} limit={u.plan.daily_tokens} unit="tokens" />
            <Card tilt={false}>
              <p className="text-sm text-muted">This month</p>
              <p className="num mt-2 text-[28px] font-medium leading-none text-ink">{(u.month.tokens + u.month.own_key_tokens).toLocaleString()}</p>
              <p className="mt-1 font-mono text-xs text-muted">tokens · {u.month.messages.toLocaleString()} answers</p>
              {u.own_key ? <p className="mt-1 font-mono text-xs text-muted">{u.month.own_key_tokens.toLocaleString()} on your own key</p> : null}
            </Card>
          </div>

          <div className="mt-4 grid gap-4 lg:grid-cols-2">
            <Routes month={u.month} />
            <Card tilt={false}>
              <h2 className="text-lg font-semibold text-ink">Research the gate skipped <span className="ml-1 chip align-middle border-amber/50! bg-amber/10!">estimate</span></h2>
              {u.estimate.available && u.estimate.saved_tokens != null ? (
                <>
                  <p className="mt-3 font-mono text-xl font-semibold text-ink">≈ {u.estimate.saved_tokens.toLocaleString()} tokens</p>
                  <p className="mt-1 text-sm text-muted">
                    {u.month.direct.toLocaleString()} answers handled directly × ≈ {u.estimate.per_escalation?.toLocaleString()} tokens,{' '}
                    {u.estimate.source === 'server'
                      ? `the average the research agent used on ${u.estimate.sample_size.toLocaleString()} escalated answers on this server in the last ${u.estimate.window_days} days.`
                      : `what research cost per question in our ${u.estimate.sample_size}-question evaluation, until this server has enough researched answers of its own.`}
                  </p>
                </>
              ) : (
                <p className="mt-3 text-sm text-muted">
                  Not enough researched answers on this server yet to estimate this ({u.estimate.sample_size} so far; at least 5 are needed).
                </p>
              )}
              <p className="mt-3 text-xs text-muted">{u.estimate.method}</p>
            </Card>
          </div>
          <p className="mt-4 text-xs text-muted">Documents stored: {u.documents} of {u.plan.documents}.</p>
        </>
      ) : null}
    </Page>
  )
}
