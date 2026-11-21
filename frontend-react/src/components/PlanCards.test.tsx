import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import type { PlanInfo } from '../lib/account'
import { PlanCards } from './PlanCards'

const PLANS: PlanInfo[] = [
  { id: 'free', label: 'Free', price_usd_month: 0, messages_per_day: 30, documents: 3, daily_tokens: 40000, concurrent_chats: 1, blurb: 'Try it.' },
  { id: 'pro', label: 'Pro', price_usd_month: 9, messages_per_day: 400, documents: 100, daily_tokens: 1000000, concurrent_chats: 3, blurb: 'Daily use.' },
]

describe('plan cards', () => {
  it('lists every limit from the plan file and the action for each plan', () => {
    render(<PlanCards plans={PLANS} action={(p) => <button>choose {p.id}</button>} />)
    expect(screen.getByText('$9')).toBeInTheDocument()
    expect(screen.getByText('400 answers a day')).toBeInTheDocument()
    expect(screen.getAllByText(/tokens a day on our keys/)).toHaveLength(2)
    expect(screen.getByText('1 answer at a time')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'choose pro' })).toBeInTheDocument()
  })
})
