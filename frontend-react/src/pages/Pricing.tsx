import { ArrowRight, Loader2 } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { PlanCards } from '../components/PlanCards'
import { ErrorNote, Page, PageHeader, SkeletonCards } from '../components/ui'
import { billingApi, type PlanInfo } from '../lib/account'
import { useApp } from '../lib/context'
import { useApi } from '../lib/useApi'

export default function Pricing() {
  const { user } = useApp()
  const plans = useApi(billingApi.plans)
  const [opening, setOpening] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [now] = useState(() => Date.now())
  const current = user?.plan ?? 'free'
  const paidUntil = user?.paid_until && user.paid_until * 1000 > now ? new Date(user.paid_until * 1000) : null

  const upgrade = async () => {
    setOpening(true)
    setError(null)
    try {
      const { url } = await billingApi.upgrade()
      window.location.assign(url)
    } catch (err) {
      setError((err as Error).message)
      setOpening(false)
    }
  }

  const action = (plan: PlanInfo) => {
    const paid = plan.price_usd_month > 0
    const onPlan = paid ? current === plan.id && paidUntil : !paidUntil
    if (!user) {
      return (
        <Link to="/signup" className="inline-flex items-center gap-2 rounded-full bg-ink px-5 py-2.5 text-sm font-semibold text-soft">
          Sign up <ArrowRight className="h-4 w-4" />
        </Link>
      )
    }
    if (onPlan) {
      return <span className={`inline-flex rounded-full border px-4 py-2 text-sm ${paid ? 'border-sky text-ice' : 'border-navy/20 text-navy'}`}>Your plan{paid && paidUntil ? ` until ${paidUntil.toLocaleDateString()}` : ''}</span>
    }
    if (!paid) return null
    return (
      <button
        onClick={upgrade}
        disabled={opening || !plans.data?.payments}
        className="inline-flex items-center gap-2 rounded-full bg-lemon px-5 py-2.5 text-sm font-semibold text-ink disabled:opacity-50"
        title={plans.data?.payments ? undefined : 'Payments are not set up on this server yet'}
      >
        {opening ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
        {paidUntil ? 'Extend' : 'Upgrade'} <ArrowRight className="h-4 w-4" />
      </button>
    )
  }

  return (
    <Page>
      <PageHeader number="12" kicker="Plans" title={<>Simple <span className="mark-lemon">pricing</span></>}>
        Start free. Pro raises every limit. Payment opens our payment provider's page; your plan is switched on once the
        payment arrives (usually within a working day), and it returns to Free if it is not renewed.
      </PageHeader>
      {error ? <div className="mb-4"><ErrorNote message={error} /></div> : null}
      {plans.error ? <ErrorNote message={plans.error} /> : null}
      {plans.loading && !plans.data ? <SkeletonCards count={2} height="h-48" /> : null}
      {plans.data ? <PlanCards plans={plans.data.plans} action={action} /> : null}
      <p className="mt-4 text-xs text-navy/60">Prices in US dollars. Limits reset at midnight UTC. Your own Groq or Gemini key, added in Settings, answers your chats without using the plan's tokens.</p>
    </Page>
  )
}
