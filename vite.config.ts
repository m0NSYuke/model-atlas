import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // The GitHub Pages build is served from /model-atlas/; local builds stay at the root.
  base: process.env.BASE_PATH ?? '/',
  plugins: [react()],
})
