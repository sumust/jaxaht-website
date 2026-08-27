import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"

// Dev: Vite on :5173 proxies /api/* to Flask on :5174.
// Prod: Flask serves the built dist/ directly.
export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        proxy: {
            "/api": {
                target: "http://localhost:5174",
                changeOrigin: true,
            },
        },
    },
})
