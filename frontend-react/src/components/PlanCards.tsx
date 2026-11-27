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

/** One panel per plan; ``action`` renders the button under each (sign up, upgrade, current). */
export function PlanCards({ plans, action }: { plans: PlanInfo[]; action: (plan: PlanInfo) => ReactNode }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {plans.map((plan) => {
        const paid = plan.price_usd_month > 0
        return (
          <Card key={plan.id} className={`flex flex-col p-6! ${paid ? 'border-t-2! border-t-clay!' : ''}`}>
            <div className="flex items-center justify-between gap-2">
              <p className="text-[15px] font-semibold text-ink">{plan.label}</p>
              {paid ? <span className="chip border-clay/40! bg-clay/10!">Most capacity</span> : null}
            </div>
            <p className="mt-4 flex items-baseline gap-1.5">
              <span className="num text-[40px] font-medium leading-none text-ink">${plan.price_usd_month}</span>
              <span className="text-sm text-muted">/ month</span>
            </p>
            {plan.blurb ? <p className="mt-2 text-sm text-muted">{plan.blurb}</p> : null}
            <ul className="mt-5 space-y-2 border-t border-line pt-5 text-sm">
              {planFeatures(plan).map((f) => (
                <li key={f} className="flex items-start gap-2.5 text-forest">
                  <Check className={`mt-0.5 h-4 w-4 shrink-0 ${paid ? 'text-clay' : 'text-muted'}`} aria-hidden /> {f}
                </li>
              ))}
            </ul>
            <div className="mt-auto pt-6">{action(plan)}</div>
          </Card>
        )
      })}
    </div>
  )
}
