// Shared Framer Motion variants: a page's panels fade in together, with a short lift.

export const stagger = {
  hidden: {},
  show: { transition: { staggerChildren: 0.04 } },
}

export const rise = {
  hidden: { opacity: 0, y: 6 },
  show: { opacity: 1, y: 0, transition: { duration: 0.28, ease: [0.22, 1, 0.36, 1] as const } },
}
