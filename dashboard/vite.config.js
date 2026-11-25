import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// base './' makes the built dist/ folder work from any host or sub-path (Netlify, Vercel, GitHub Pages, ...)
// chunkSizeWarningLimit: the bundle includes the ~400 KB trained model weights (siren_model.json)
export default defineConfig({ base: './', plugins: [react(), tailwindcss()], build: { chunkSizeWarningLimit: 800 } })
