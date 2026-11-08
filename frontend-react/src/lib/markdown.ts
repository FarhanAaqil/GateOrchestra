/**
 * Prepares model Markdown for rendering.
 *
 * Models write math as \( … \) and \[ … \]; remark-math expects dollars. Single
 * dollars stay plain text because they usually mean money ("$5,000 and $11,260"),
 * so inline math becomes $$…$$ and display math a $$ block. Code is left alone.
 */
export function normalizeMath(text: string): string {
  const parts = text.split(/(```[\s\S]*?```|`[^`\n]*`)/g)
  return parts
    .map((part, i) => {
      if (i % 2 === 1) return part // inside code
      return part
        .replace(/\\\[([\s\S]+?)\\\]/g, (_, m: string) => `\n$$\n${m.trim()}\n$$\n`)
        .replace(/\\\(([\s\S]+?)\\\)/g, (_, m: string) => `$$${m.trim()}$$`)
    })
    .join('')
}
