import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    // three.js is large by nature; it is split into its own lazily loaded chunk.
    chunkSizeWarningLimit: 1200,
  },
})
