import { AlertTriangle } from 'lucide-react'
import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Card, Page, PageHeader } from '../components/ui'

/**
 * Starting drafts for a hosted GateOrchestra. They describe what the code actually
 * does with data, but they are not legal advice: a person qualified for the
 * operator's country must review and complete them (the [bracketed] parts) first.
 */
function Draft({ number, title, children }: { number: string; title: string; children: ReactNode }) {
  return (
    <Page>
      <PageHeader number={number} kicker="Legal" title={title} />
      <div className="mb-5 flex items-start gap-2 rounded-lg border-2 border-wine bg-wine/10 px-4 py-3 text-sm text-wine" role="note">
        <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
        <p>
          <strong>DRAFT: needs human review.</strong> This text is a starting point written from how the software works. It is not
          legal advice and has not been checked by a lawyer. Parts in [brackets] must be filled in by the operator.
        </p>
      </div>
      <Card tilt={false} className="prose-legal max-w-3xl space-y-4 text-sm leading-relaxed text-navy">
        {children}
      </Card>
    </Page>
  )
}

function H({ children }: { children: ReactNode }) {
  return <h2 className="pt-2 text-lg font-semibold text-ink">{children}</h2>
}

export function Terms() {
  return (
    <Draft number="13" title="Terms of service">
      <p>These terms cover your use of GateOrchestra at [domain], run by [operator name, address] ("we"). By creating an account you accept them.</p>
      <H>The service</H>
      <p>GateOrchestra is an AI assistant. Answers are written by large language models and can be wrong, incomplete or out of date. Check anything important yourself; do not rely on answers for medical, legal, financial or safety decisions.</p>
      <H>Your account</H>
      <p>You need a confirmed email address. Keep your password to yourself; you are responsible for what happens under your account. You must be at least [16] years old.</p>
      <H>Plans and payment</H>
      <p>The Free plan costs nothing. Pro is paid per month in advance through our payment provider. Your plan is switched on once we receive the payment and ends when the paid period ends unless you pay again; there is no automatic renewal. [Refund policy.] Limits for each plan are shown on the <Link to="/pricing" className="underline">pricing page</Link> and may change with [30] days' notice for paid periods already bought.</p>
      <H>Your content</H>
      <p>You keep the rights to what you write and upload. You give us permission to store and process it only to run the service for you, including sending it to the model providers listed in the <Link to="/privacy" className="underline">privacy notice</Link>. You must have the right to upload what you upload.</p>
      <H>Acceptable use</H>
      <p>You agree to the <Link to="/acceptable-use" className="underline">acceptable-use policy</Link>. We may suspend or close accounts that break it.</p>
      <H>Ending</H>
      <p>You can delete your account at any time in Settings. We may end the service with [30] days' notice, giving you time to export your data.</p>
      <H>Liability</H>
      <p>[The service is provided as is. Limitation of liability, governing law and jurisdiction, to be written for the operator's country.]</p>
      <p className="text-xs text-navy/60">Contact: [contact email]. Last updated: [date].</p>
    </Draft>
  )
}

export function Privacy() {
  return (
    <Draft number="14" title="Privacy notice">
      <p>[Operator name, address, contact email] is responsible for your data on this service.</p>
      <H>What we keep</H>
      <ul className="list-disc space-y-1 pl-5">
        <li><strong>Account:</strong> your email address, a one-way (argon2id) hash of your password, when you confirmed your email, your plan and how long it is paid for.</li>
        <li><strong>Sign-in:</strong> one session cookie, which is needed for the site to work, and a hash of its value on our side. No advertising or tracking cookies.</li>
        <li><strong>What you write:</strong> your conversations, the text extracted from documents you upload, your ratings of answers, and cached answers to repeated questions (kept for [24 hours]).</li>
        <li><strong>Usage counts:</strong> per day, how many answers you received, how many tokens they used and how each was routed. No content.</li>
        <li><strong>Your own API keys</strong>, if you add any, encrypted.</li>
        <li><strong>Server logs:</strong> for each request, the time, a request id, your account id, the address path, the result and how long it took. Never what you wrote.</li>
      </ul>
      <H>Who else receives data</H>
      <ul className="list-disc space-y-1 pl-5">
        <li>Your questions, the conversation so far and relevant document passages are sent to the AI model providers that write answers: Groq and Google (Gemini). Their own terms apply to that processing. [Check each provider's data-retention and training terms.]</li>
        <li>When the assistant researches a question it sends search terms to Wikipedia.</li>
        <li>Our email provider [name] sends account emails; our payment provider [name] handles payments, and we never see your card details.</li>
        <li>[If error tracking is used: error reports go to Sentry without request data, cookies or content.]</li>
        <li>Our hosting provider [name] runs the server in [country].</li>
      </ul>
      <H>How long</H>
      <p>Until you delete it or your account. Backups of the database are kept for [14] days, so deleted data disappears from them within that time.</p>
      <H>Your choices</H>
      <p>In <Link to="/settings" className="underline">Settings</Link> you can export your conversations and delete your account, which removes your conversations, documents, cached answers, ratings, keys and usage history. [Describe the rights that apply in the operator's country (for example access, correction, objection, complaint to a regulator) and the legal basis for processing.]</p>
      <p className="text-xs text-navy/60">Last updated: [date].</p>
    </Draft>
  )
}

export function AcceptableUse() {
  return (
    <Draft number="15" title="Acceptable-use policy">
      <p>To keep the service working for everyone, do not use GateOrchestra to:</p>
      <ul className="list-disc space-y-1 pl-5">
        <li>break the law, or help anyone else break it;</li>
        <li>create or spread malware, phishing, spam, or content that sexualises minors;</li>
        <li>harass, threaten or impersonate people, or collect their personal data without a right to;</li>
        <li>try to make the assistant ignore its instructions, get round the safety screen, or escape or overload the code sandbox;</li>
        <li>get round plan limits, for example with several accounts or by sharing one account;</li>
        <li>probe, scan or attack the service or other people's accounts;</li>
        <li>scrape the service or resell access without our written permission.</li>
      </ul>
      <p>We may refuse a message, limit an account, or close it for breaking this policy. [Describe how people can report abuse and appeal: contact email.]</p>
      <p className="text-xs text-navy/60">Last updated: [date].</p>
    </Draft>
  )
}
