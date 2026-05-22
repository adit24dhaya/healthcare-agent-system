import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#080a0d",
        panel: "#101318",
        panelMuted: "#151922",
        borderSoft: "rgba(255,255,255,0.08)",
        ink: "#f4f6f8",
        muted: "#9aa4b2",
        success: "#2f9b6a",
        warning: "#d49a35",
        danger: "#d05245",
        info: "#5aa3d9",
      },
      boxShadow: {
        console: "0 18px 60px rgba(0, 0, 0, 0.35)",
      },
    },
  },
  plugins: [],
};

export default config;
