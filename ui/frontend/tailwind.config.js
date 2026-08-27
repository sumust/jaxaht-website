/** @type {import('tailwindcss').Config} */
export default {
    content: ["./index.html", "./src/**/*.{ts,tsx}"],
    theme: {
        extend: {
            colors: {
                // Hanabi color palette. CSS color-mix safe, colorblind-friendly-ish.
                hanabi: {
                    red:    "#e85d75",
                    yellow: "#f4c95d",
                    green:  "#6bbf59",
                    white:  "#f1f5f9",
                    blue:   "#60a5fa",
                },
                // UT Austin burnt orange — accent color for LARG branding.
                "ut-orange": "#BF5700",
                // Neutral surface scale (near-black to off-white).
                ink: {
                    50:  "#fafafa",
                    100: "#f4f4f5",
                    200: "#e4e4e7",
                    300: "#d4d4d8",
                    400: "#a1a1aa",
                    500: "#71717a",
                    600: "#52525b",
                    700: "#3f3f46",
                    800: "#27272a",
                    900: "#18181b",
                    950: "#09090b",
                },
            },
            fontFamily: {
                sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "sans-serif"],
                mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
                display: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
            },
            animation: {
                "slide-up":  "slideUp 200ms ease-out",
                "fade-in":   "fadeIn 120ms ease-out",
                "card-flip": "cardFlip 400ms cubic-bezier(0.4, 0, 0.2, 1)",
            },
            keyframes: {
                slideUp:  { "0%": { transform: "translateY(4px)", opacity: "0" }, "100%": { transform: "translateY(0)", opacity: "1" } },
                fadeIn:   { "0%": { opacity: "0" }, "100%": { opacity: "1" } },
                cardFlip: { "0%": { transform: "rotateY(90deg)" }, "100%": { transform: "rotateY(0)" } },
            },
        },
    },
    plugins: [],
}
