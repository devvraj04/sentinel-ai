/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        navy:   { DEFAULT: "#1B2A4A", 50: "#EBF0FA" },
        risk: {
          green:  "#16A34A",
          yellow: "#D97706",
          orange: "#EA580C",
          red:    "#DC2626",
        }
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
}
