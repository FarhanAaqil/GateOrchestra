import 'highlight.js/styles/github-dark.css'
import 'katex/dist/katex.min.css'
import { Check, Copy } from 'lucide-react'
import { memo, useState, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeHighlight from 'rehype-highlight'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import { normalizeMath } from '../lib/markdown'

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node)
  if (Array.isArray(node)) return node.map(textOf).join('')
  if (node && typeof node === 'object' && 'props' in node) {
    return textOf((node as { props: { children?: ReactNode } }).props.children)
  }
  return ''
}

function CodeBlock({ children }: { children?: ReactNode }) {
  const [copied, setCopied] = useState(false)
  const code = textOf(children).replace(/\n$/, '')
  const lang = /language-(\w+)/.exec(
    ((children as { props?: { className?: string } } | undefined)?.props?.className) ?? '',
  )?.[1]
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard can be blocked (insecure context); nothing to do.
    }
  }
  return (
    <div className="group relative my-3 overflow-hidden rounded-xl bg-[#0d1117]">
      <div className="flex items-center justify-between border-b border-white/10 px-3 py-1.5 font-mono text-[11px] text-sky">
        <span>{lang ?? 'code'}</span>
        <button onClick={copy} className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 hover:bg-white/10" aria-label="Copy code">
          {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-[13px] leading-relaxed">{children}</pre>
    </div>
  )
}

/** Renders an assistant answer: GitHub-flavoured Markdown with highlighted code. */
export const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <div className="prose-chat">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, [remarkMath, { singleDollarTextMath: false }]]}
        rehypePlugins={[[rehypeKatex, { throwOnError: false, strict: 'ignore' }], [rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={{
          pre: ({ children }) => <CodeBlock>{children}</CodeBlock>,
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer noopener" className="font-medium text-wine underline underline-offset-2">
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="my-3 overflow-x-auto">
              <table className="w-full border-collapse text-sm">{children}</table>
            </div>
          ),
        }}
      >
        {normalizeMath(text)}
      </ReactMarkdown>
    </div>
  )
})
