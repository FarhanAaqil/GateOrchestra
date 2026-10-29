import { AnimatePresence } from 'framer-motion'
import { lazy, Suspense } from 'react'
import { Route, Routes, useLocation } from 'react-router-dom'
import Layout from './components/Layout'
import { Spinner } from './components/ui'

// Pages are code-split so each loads only its own charts.
const Overview = lazy(() => import('./pages/Overview'))
const RunTask = lazy(() => import('./pages/RunTask'))
const GateInspector = lazy(() => import('./pages/GateInspector'))
const Benchmark = lazy(() => import('./pages/Benchmark'))
const Baselines = lazy(() => import('./pages/Baselines'))
const Results = lazy(() => import('./pages/Results'))
const TokenAccountant = lazy(() => import('./pages/TokenAccountant'))
const About = lazy(() => import('./pages/About'))

export default function App() {
  const location = useLocation()
  return (
    <Layout>
      <Suspense fallback={<Spinner label="Loading page" />}>
        <AnimatePresence mode="wait">
          <Routes location={location} key={location.pathname}>
            <Route path="/" element={<Overview />} />
            <Route path="/run" element={<RunTask />} />
            <Route path="/gate" element={<GateInspector />} />
            <Route path="/benchmark" element={<Benchmark />} />
            <Route path="/baselines" element={<Baselines />} />
            <Route path="/results" element={<Results />} />
            <Route path="/tokens" element={<TokenAccountant />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Overview />} />
          </Routes>
        </AnimatePresence>
      </Suspense>
    </Layout>
  )
}
