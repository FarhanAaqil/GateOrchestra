import type { ResultMethod, Strategy } from './types'

// Palette (see index.css): only these ten colours appear in the UI.
export const PALETTE = {
  paper: '#FFFFFF',
  sage: '#CFE3EC',
  /** Sage dark enough to read as a mark on a light card. */
  sageDeep: '#5F9E7F',
  clay: '#8B1B1B',
  forest: '#0B4A6B',
  canvas: '#FAF7F0',
  butter: '#F5E6C8',
  wine: '#8B1B1B',
  saffron: '#B07D2B',
  amber: '#9A6414',
  ink: '#1A1D1C',
} as const

export const METHOD_LABELS: Record<ResultMethod, string> = {
  'CoT-SC-only': 'CoT-SC only',
  'Always-MAS': 'Always-MAS',
  RandomGate: 'Random gate',
  RuleBasedGate: 'Rule-based gate',
  GateOrchestra: 'GateOrchestra',
}

export const METHOD_COLORS: Record<ResultMethod, string> = {
  'CoT-SC-only': PALETTE.sageDeep,
  'Always-MAS': PALETTE.wine,
  RandomGate: PALETTE.forest,
  RuleBasedGate: PALETTE.amber,
  GateOrchestra: PALETTE.clay,
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
