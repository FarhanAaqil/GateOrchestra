import { AnimatePresence } from 'framer-motion'
import { lazy, Suspense } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import { AdminOnly, RequireUser } from './components/Guards'
import Layout from './components/Layout'
import { Spinner } from './components/ui'
import { useApp } from './lib/context'

// Pages are code-split so each loads only its own charts.
const Overview = lazy(() => import('./pages/Overview'))
const Chat = lazy(() => import('./pages/Chat'))
const RunTask = lazy(() => import('./pages/RunTask'))
const GateInspector = lazy(() => import('./pages/GateInspector'))
const Benchmark = lazy(() => import('./pages/Benchmark'))
const Baselines = lazy(() => import('./pages/Baselines'))
const Results = lazy(() => import('./pages/Results'))
const TokenAccountant = lazy(() => import('./pages/TokenAccountant'))
const About = lazy(() => import('./pages/About'))
const Admin = lazy(() => import('./pages/Admin'))
const Settings = lazy(() => import('./pages/Settings'))
const Usage = lazy(() => import('./pages/Usage'))
const Landing = lazy(() => import('./pages/Landing'))
const Login = lazy(() => import('./pages/Auth').then((m) => ({ default: m.Login })))
const Signup = lazy(() => import('./pages/Auth').then((m) => ({ default: m.Signup })))
const ForgotPassword = lazy(() => import('./pages/Auth').then((m) => ({ default: m.ForgotPassword })))
const ResetPassword = lazy(() => import('./pages/Auth').then((m) => ({ default: m.ResetPassword })))
const VerifyEmail = lazy(() => import('./pages/Auth').then((m) => ({ default: m.VerifyEmail })))

/** A hosted server greets visitors with the product page; the research overview moves to /overview. */
function Home() {
  const { hosted } = useApp()
  return hosted ? <Landing /> : <Overview />
}

export default function App() {
  const location = useLocation()
  return (
    <Layout>
      <Suspense fallback={<Spinner label="Loading page" />}>
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<Home />} />
            <Route path="/overview" element={<Overview />} />
            <Route path="/chat" element={<RequireUser><Chat /></RequireUser>} />
            <Route path="/run" element={<AdminOnly><RunTask /></AdminOnly>} />
            <Route path="/gate" element={<GateInspector />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/baselines" element={<Baselines />} />
            <Route path="/results" element={<Results />} />
            <Route path="/tokens" element={<AdminOnly><TokenAccountant /></AdminOnly>} />
            <Route path="/about" element={<About />} />
            <Route path="/admin" element={<AdminOnly><Admin /></AdminOnly>} />
            <Route path="/usage" element={<RequireUser><Usage /></RequireUser>} />
            <Route path="/settings" element={<RequireUser><Settings /></RequireUser>} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
            <Route path="/forgot" element={<ForgotPassword />} />
            <Route path="/reset" element={<ResetPassword />} />
            <Route path="/verify" element={<VerifyEmail />} />
            <Route path="*" element={<Overview />} />
          </Routes>
        </AnimatePresence>
      </Suspense>
    </Layout>
  )
}
