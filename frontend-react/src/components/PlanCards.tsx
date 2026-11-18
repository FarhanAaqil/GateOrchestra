import { Check } from 'lucide-react'
import type { ReactNode } from 'react'
import type { PlanInfo } from '../lib/account'
import { Card } from './ui'

function planFeatures(plan: PlanInfo): string[] {
  return [
    `${plan.messages_per_day.toLocaleString()} answers a day`,
    `${plan.daily_tokens.toLocaleString()} tokens a day on our keys`,
    `${plan.documents.toLocaleString()} stored document${plan.documents === 1 ? '' : 's'}`,
    `${plan.concurrent_chats} answer${plan.concurrent_chats === 1 ? '' : 's'} at a time`,
    'Bring your own Groq or Gemini key: its tokens do not count',
  ]
}

/** One card per plan; ``action`` renders the button under each (sign up, upgrade, current). */
export function PlanCards({ plans, action }: { plans: PlanInfo[]; action: (plan: PlanInfo) => ReactNode }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {plans.map((plan) => {
        const paid = plan.price_usd_month > 0
        return (
          <Card key={plan.id} navy={paid} tilt={false}>
            <p className={`font-mono text-xs uppercase tracking-widest ${paid ? 'text-cyan' : 'text-wine'}`}>{plan.label}</p>
            <p className={`mt-3 text-4xl font-bold ${paid ? 'text-ice' : 'text-ink'}`}>
              {paid ? `$${plan.price_usd_month}` : 'Free'}
              {paid ? <span className="text-base font-normal opacity-70"> / month</span> : null}
            </p>
            {plan.blurb ? <p className={`mt-2 text-sm ${paid ? 'text-sky' : 'text-navy/70'}`}>{plan.blurb}</p> : null}
            <ul className="mt-4 space-y-1.5 text-sm">
              {planFeatures(plan).map((f) => (
                <li key={f} className={`flex items-start gap-2 ${paid ? 'text-ice' : 'text-navy'}`}>
                  <Check className={`mt-0.5 h-4 w-4 shrink-0 ${paid ? 'text-lemon' : 'text-wine'}`} aria-hidden /> {f}
                </li>
              ))}
            </ul>
            <div className="mt-5">{action(plan)}</div>
          </Card>
        )
      })}
    </div>
  )
}
