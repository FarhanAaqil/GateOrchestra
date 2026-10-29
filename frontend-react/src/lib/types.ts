// Shapes returned by the GateOrchestra API (api/main.py). Optional fields are
// absent for some methods; nothing here is invented client-side.

export type MethodId = 'GateOrchestra' | 'RuleBasedGate' | 'RandomGate' | 'Always-MAS' | 'CoT-SC'
export type ResultMethod =
  | 'CoT-SC-only'
  | 'Always-MAS'
  | 'RandomGate'
  | 'RuleBasedGate'
  | 'GateOrchestra'
export type Strategy = 'react' | 'debate' | 'reflexion'
export type ResultsStatus = 'verified' | 'partial' | 'unverified' | 'no_results'

export interface GateInfo {
  path: string
  seed: number | null
  classifier: string
  degenerate: boolean
  build_info: { feature_backend: string; sklearn_version: string } | null
}

export interface ModelsResponse {
  methods: { id: MethodId; name: string; type: string; description: string; available: boolean }[]
  providers: string[]
  current_provider: string
  default_model: string
  trained_gate_available: boolean
  trained_gate: GateInfo | null
  /** Completion-token budget for one whole CoT-SC probe (shared/config.py). */
  probe_token_budget: number
  /** Escalation multipliers swept during calibration. */
  k_values: number[]
}

export interface Task {
  task_id: string
  question: string
  context: string | null
  depth_score: number | null
  parallel_score: number | null
  ground_truth: string | null
  answer_aliases: string[]
  source_dataset: string | null
}

export interface TasksResponse {
  split: 'train' | 'val' | 'test'
  total: number
  limit: number
  offset: number
  tasks: Task[]
}

export interface GateDecision {
  task_id: string
  decision: 'STOP' | 'ESCALATE'
  confidence: number
  token_budget_cap: number | null
  gate_type: string
}

export interface GateFeatures {
  task_id: string
  consistency_score: number
  probe_tokens: number
  question_word_count: number
  entity_count: number
  clause_count: number
  has_context: boolean
  estimated_depth: number | null
  estimated_parallel: number | null
}

export interface ProbeDetails {
  answer: string
  consistency_score: number
  tokens_used: number
  latency_ms: number | null
  samples: string[]
}

export interface TraceStep {
  step: string
  output: string
  tokens: number
}

export interface RunResult {
  task_id: string
  method: ResultMethod
  predicted_answer: string
  is_correct: boolean | null
  tokens_spent: number
  probe_tokens: number | null
  mas_tokens: number | null
  gate_decision: GateDecision | null
  latency_ms: number | null
  mas_strategy: Strategy | null
  simulated: boolean
  features?: GateFeatures
  probe?: ProbeDetails | null
  trace?: TraceStep[]
}

export interface RunRequest {
  task_id: string
  question: string
  context?: string | null
  ground_truth?: string | null
  method: MethodId
  k: number
  force_simulation: boolean
  threshold?: number | null
}

export interface AggregatedRow {
  method: ResultMethod
  seeds: number[]
  n_tasks_per_seed?: number[]
  accuracy_mean: number
  accuracy_std: number
  accuracy_pooled_ci95?: [number, number]
  avg_tokens_mean: number
  token_savings_pct_mean: number
  escalate_rate_mean?: number
  avg_latency_ms_mean?: number
  n_seeds: number
}

export interface PairedTest {
  method: string
  reference: string
  n_pairs: number
  method_only_correct: number
  reference_only_correct: number
  p_value: number
}

export interface SummaryResponse {
  status: ResultsStatus
  message?: string
  verification_issues?: string[]
  seeds?: number[]
  n_per_split?: number | null
  aggregated: AggregatedRow[]
  pooled_paired_tests?: PairedTest[]
}

export interface ImportanceResponse {
  available: boolean
  reason?: string
  gate: GateInfo | null
  importances: { feature: string; importance: number }[]
}

export interface PerTaskRow {
  seed: number
  method: ResultMethod
  task_id: string
  is_correct: boolean | null
  tokens_spent: number
  decision: 'STOP' | 'ESCALATE' | null
  depth: number | null
  parallel: number | null
  source: string | null
}

export interface ResultsDetail {
  status: ResultsStatus
  per_task: PerTaskRow[]
  axis_cells: {
    method: ResultMethod
    depth: number | null
    parallel: number | null
    n: number
    accuracy: number
    escalation_rate: number
    avg_tokens: number
  }[]
  escalation_quality: {
    method: ResultMethod
    true_positive: number
    false_positive: number
    false_negative: number
    precision: number | null
    recall: number | null
  }[]
  k_sweep: {
    seed: number
    k: number
    accuracy: number
    avg_tokens: number
    token_savings_pct: number
    escalation_rate: number
    selected: boolean
  }[]
  gates: { seed: number; classifier?: string; tau_acc?: number; degenerate?: boolean }[]
}
