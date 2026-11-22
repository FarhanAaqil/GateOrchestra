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
        <Link to="/signup" className="btn btn-primary w-full">
          Sign up <ArrowRight className="h-4 w-4" />
        </Link>
      )
    }
    if (onPlan) {
      return <span className="btn btn-secondary w-full cursor-default">Your plan{paid && paidUntil ? ` until ${paidUntil.toLocaleDateString()}` : ''}</span>
    }
    if (!paid) return null
    if (!plans.data?.payments) {
      // No payment page configured: say so on the page instead of showing a dead button.
      return user.is_admin ? (
        <div className="space-y-2">
          <Link to="/admin" className="btn btn-primary w-full">
            Give an account Pro in Admin <ArrowRight className="h-4 w-4" />
          </Link>
          <p className="text-xs text-forest/70">
            Payments are not set up yet. Add your payment page as <code className="font-mono">GATE_PAYMENT_LINK</code> in
            .env to turn on the Upgrade button for everyone.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          <span className="btn btn-secondary w-full cursor-default">Upgrades open soon</span>
          <p className="text-xs text-forest/70">Paid plans are not on sale yet. The free plan has everything you need to try it.</p>
        </div>
      )
    }
    return (
      <button onClick={upgrade} disabled={opening} className="btn btn-accent w-full">
        {opening ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
        {paidUntil ? 'Extend' : 'Upgrade'} <ArrowRight className="h-4 w-4" />
      </button>
    )
  }

  return (
    <Page>
      <PageHeader kicker="Plans" title="Simple pricing">
        Start free. Pro raises every limit. Payment opens our payment provider's page; your plan is switched on once the
        payment arrives (usually within a working day), and it returns to Free if it is not renewed.
      </PageHeader>
      {error ? <div className="mb-4"><ErrorNote message={error} /></div> : null}
      {plans.error ? <ErrorNote message={plans.error} /> : null}
      {plans.loading && !plans.data ? <SkeletonCards count={2} height="h-48" /> : null}
      {plans.data ? <PlanCards plans={plans.data.plans} action={action} /> : null}
      <p className="mt-4 text-xs text-forest/70">Prices in US dollars. Limits reset at midnight UTC. Your own Groq or Gemini key, added in Settings, answers your chats without using the plan's tokens.</p>
    </Page>
  )
}
