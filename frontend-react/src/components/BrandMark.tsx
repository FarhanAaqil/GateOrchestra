// The mark is the gate itself, drawn as a plain geometric glyph: two uprights,
// a lintel, and the burgundy diamond on the threshold that is the decision.
// A single question enters on the left; the few that escalate leave on the right.
// public/favicon.svg is the same drawing.
export function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" className="shrink-0" aria-hidden>
      <rect width="32" height="32" rx="8" fill="var(--color-forest)" />
      <path d="M9 23V11.5A2.5 2.5 0 0 1 11.5 9h9A2.5 2.5 0 0 1 23 11.5V23" fill="none" stroke="#ffffff" strokeWidth="2.1" strokeLinecap="round" />
      <path d="M5.5 23h21" stroke="var(--color-sage)" strokeWidth="2.1" strokeLinecap="round" />
      <path d="M16 13.2l3 3.1-3 3.1-3-3.1z" fill="var(--color-clay-light)" />
    </svg>
  )
}

/**
 * The name as a wordmark: "Gate" in the interface face, "Orchestra" in the serif
 * italic. Colour is inherited, so it reads on cream and on the teal sidebar alike.
 */
export function Wordmark({ className = '' }: { className?: string }) {
  return (
    <span className={`text-[17px] leading-none tracking-tight ${className}`}>
      <span className="font-semibold">Gate</span>
      <span className="font-display font-medium italic">Orchestra</span>
    </span>
  )
}
