# GateOrchestra console

React 19 + TypeScript + Vite, styled with Tailwind v4, 3D with three.js via
@react-three/fiber, drei and postprocessing, animation with Framer Motion, and
charts with Recharts.

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check friendly production build in dist/
npx tsc --noEmit   # type check
npm run lint
```

The API address defaults to `http://127.0.0.1:8000`; set `VITE_API_BASE_URL` in
`.env.local` to point elsewhere.

## Layout

| Folder | Contents |
| :--- | :--- |
| `src/pages/` | One file per page: Overview, Run a Task, Gate Inspector, Benchmark, Baselines, Results, Token Accountant, About |
| `src/three/` | The gate scene (3D and its SVG twin share `gateTimeline.ts`), the neural-field background |
| `src/charts/` | Token gauge, confidence gauge, Recharts charts, Depth × Parallel heatmap |
| `src/components/` | Layout (sidebar, top bar) and shared UI pieces |
| `src/lib/` | Typed API client, app state, formatting, palette |

There is no mock data. Every number comes from the API; pages without verified
benchmark results show an empty state instead.

## Palette

Only ten colours are used (see `src/index.css`): Ice `#F1FAFF`, Sky `#A8DBFF`,
Ember `#FF6B35`, Deep Navy `#1B263B`, Soft White `#F7F7F7`, Lemon `#FFF56B`,
Wine `#8B1538`, Electric Cyan `#00E5FF`, Tangerine `#FF8C42`, Ink `#1B1B1F`.
In the pipeline: cyan is the gate, sky the CoT-SC path, ember/wine the
multi-agent path, tangerine the token budget, wine over-budget.

## Motion and performance

3D scenes are lazy-loaded. They are replaced by a 2D SVG version on screens
under 768px, when the OS asks for reduced motion, when WebGL is unavailable, or
when **Performance** mode is on (top bar).

## Installing inside Google Drive

`npm install` fails with `EBADF` when the project lives in a Google Drive
folder (Drive cannot keep up with npm's parallel writes). Install in a local
folder and mirror the result:

```powershell
$local = "$env:USERPROFILE\.gateorchestra-build\frontend"
New-Item -ItemType Directory -Force $local | Out-Null
Copy-Item package.json, package-lock.json $local
Push-Location $local; npm ci; Pop-Location
robocopy "$local\node_modules" .\node_modules /MIR /MT:1 /R:3 /W:1 /NFL /NDL
```
