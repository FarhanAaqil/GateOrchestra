// The mark is a G drawn as the quick check going round. Its crossbar is the quick path
// (sage), the saffron dot is the gate, and the clay route leaves through the opening of
// the G: the few questions that escalate. public/favicon.svg is the same drawing.
export function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" className="shrink-0" aria-hidden>
      <rect width="32" height="32" rx="9" fill="var(--color-forest)" />
      <path d="M19.8 7.6A9.2 9.2 0 1 0 25.2 16.6" fill="none" stroke="var(--color-paper)" strokeWidth="3" strokeLinecap="round" />
      <path d="M16.6 16.6H25.2" stroke="var(--color-sage)" strokeWidth="3" strokeLinecap="round" />
      <path d="M16.6 16.6C19.6 16.6 21 10.8 24.6 8.2" fill="none" stroke="var(--color-clay)" strokeWidth="2.4" strokeLinecap="round" />
      <circle cx="16.6" cy="16.6" r="2.5" fill="var(--color-saffron)" />
      <circle cx="25.6" cy="7.4" r="2.6" fill="var(--color-clay)" />
    </svg>
  )
}

/** The name as a wordmark: "Gate" in the interface face, "Orchestra" in the serif italic. */
export function Wordmark({ className = '' }: { className?: string }) {
  return (
    <span className={`text-[16px] leading-none tracking-tight text-ink ${className}`}>
      <span className="font-semibold">Gate</span>
      <span className="font-display font-medium italic">Orchestra</span>
    </span>
  )
}
