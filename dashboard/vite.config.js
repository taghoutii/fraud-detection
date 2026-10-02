import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      // charting library (and the React runtime it pulls in) in its own long-cacheable chunk
      output: { manualChunks: { vendor: ["recharts", "react", "react-dom"] } },
    },
    chunkSizeWarningLimit: 600, // the vendor chunk is ~550 kB raw / ~155 kB gzipped; app code is ~50 kB
  },
});
