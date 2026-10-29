import type { ResultMethod, Strategy } from './types'

// Palette (see index.css): only these ten colours appear in the UI.
export const PALETTE = {
  ice: '#F1FAFF',
  sky: '#A8DBFF',
  ember: '#FF6B35',
  navy: '#1B263B',
  soft: '#F7F7F7',
  lemon: '#FFF56B',
  wine: '#8B1538',
  cyan: '#00E5FF',
  tangerine: '#FF8C42',
  ink: '#1B1B1F',
} as const

export const METHOD_LABELS: Record<ResultMethod, string> = {
  'CoT-SC-only': 'CoT-SC only',
  'Always-MAS': 'Always-MAS',
  RandomGate: 'Random gate',
  RuleBasedGate: 'Rule-based gate',
  GateOrchestra: 'GateOrchestra',
}

export const METHOD_COLORS: Record<ResultMethod, string> = {
  'CoT-SC-only': PALETTE.sky,
  'Always-MAS': PALETTE.wine,
  RandomGate: PALETTE.navy,
  RuleBasedGate: PALETTE.tangerine,
  GateOrchestra: PALETTE.ember,
}

export const STRATEGY_LABELS: Record<Strategy, string> = {
  react: 'ReAct',
  debate: 'Debate',
  reflexion: 'Reflexion',
}

export const FEATURE_LABELS: Record<string, string> = {
  consistency_score: 'Probe consistency',
  probe_tokens: 'Probe tokens',
  question_word_count: 'Question words',
  entity_count: 'Entities',
  clause_count: 'Clauses',
  has_context: 'Has context',
  estimated_depth: 'Est. depth',
  estimated_parallel: 'Est. parallel',
}

export const pct = (value: number | null | undefined, digits = 1): string =>
  value == null ? '—' : `${value.toFixed(digits)}%`

export const num = (value: number | null | undefined, digits = 0): string =>
  value == null ? '—' : value.toLocaleString(undefined, { maximumFractionDigits: digits })

/** Results with status unverified/no_results are not shown as findings. */
export const isCitable = (status: string | undefined): boolean =>
  status === 'verified' || status === 'partial'
