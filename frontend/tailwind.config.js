/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0a0f1a',
        card: '#0f172a',
        border: '#1e293b',
        accent: '#14b8a6',
      }
    },
  },
  plugins: [],
}
