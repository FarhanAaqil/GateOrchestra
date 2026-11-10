/** Plain-language meanings for the terms the Results page uses. Shown as hover text. */
export const GLOSSARY = {
  precision: 'Of the tasks the gate sent to the agent team, the share that really needed it (the agent team was right and the quick check was wrong).',
  recall: 'Of the tasks that really needed the agent team, the share the gate actually sent to it.',
  tp: 'True positives: tasks the gate escalated that needed the agent team.',
  fp: 'False positives: tasks the gate escalated that did not need the agent team (wasted tokens).',
  fn: 'False negatives: tasks that needed the agent team but the gate kept the quick answer.',
  tauAcc: 'The accuracy gap used to label training tasks as "needs the agent team". Changing it means retraining the gate.',
  k: 'The budget multiplier: the agent team may spend up to k times the tokens the quick check used.',
} as const

export type GlossaryTerm = keyof typeof GLOSSARY
