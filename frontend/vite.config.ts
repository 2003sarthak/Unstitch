import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    // Fail rather than quietly moving to 5174. The backend allows exactly one
    // origin via CORS_ORIGINS, so a silent port change turns into a confusing
    // browser-only CORS failure that looks like a backend bug.
    strictPort: true,
  },
});
