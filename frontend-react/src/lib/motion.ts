// Shared Framer Motion variants: pages stagger their children, cards rise in.

export const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07 } },
}

export const rise = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0, transition: { duration: 0.45, ease: [0.22, 1, 0.36, 1] as const } },
}
