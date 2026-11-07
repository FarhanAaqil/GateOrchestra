# Final deck

`GateOrchestra_final.pptx` is generated from the real results, so its numbers
always match `results/real/final/master_results.json` and
`reports/final_analysis.json`. The talking points that depend on the findings
live in `narrative.json`.

```bash
python scripts/analyze_final_results.py
npm install pptxgenjs@3.12.0          # once, outside Google Drive
node reports/deck/build_deck.js
```
