import { Link, useLocation } from "react-router-dom"
import type { ReactNode } from "react"

type Props = {
    children: ReactNode
    title?: string
    subtitle?: ReactNode
    right?: ReactNode
}

export function Layout({ children, title, subtitle, right }: Props) {
    const location = useLocation()
    const isHome = location.pathname === "/"

    return (
        <div className="min-h-screen flex flex-col">
            <header className="border-b border-ink-800 bg-ink-950/80 backdrop-blur-sm sticky top-0 z-10">
                <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
                    <Link to="/" className="flex items-center gap-2 group">
                        <span className="text-lg font-bold tracking-tight">
                            <span className="text-ut-orange">Jax</span>
                            <span className="text-ink-100">AHT</span>
                        </span>
                    </Link>
                    <nav className="flex items-center gap-2 text-sm">
                        {!isHome && <Link to="/" className="btn-ghost">Home</Link>}
                        <Link to="/about" className="btn-ghost">About</Link>
                        <a
                            href="https://github.com/LARG/jax-aht"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="btn-ghost"
                        >
                            GitHub
                        </a>
                    </nav>
                </div>
            </header>

            <main className="flex-1 max-w-7xl w-full mx-auto px-4 py-8">
                {title && (
                    <div className="mb-8 flex items-start justify-between gap-4">
                        <div>
                            <h1 className="text-2xl font-bold tracking-tight text-balance">
                                {title}
                            </h1>
                            {subtitle && (
                                <p className="text-sm text-ink-400 mt-1">{subtitle}</p>
                            )}
                        </div>
                        {right && <div>{right}</div>}
                    </div>
                )}
                {children}
            </main>

            <footer className="border-t border-ink-800 text-xs text-ink-500 py-4 mt-auto">
                <div className="max-w-7xl mx-auto px-4 flex items-center justify-between">
                    <div><span className="text-ut-orange font-medium">JaxAHT</span> · UT Austin</div>
                    <div className="flex items-center gap-4">
                        <a
                            href="https://github.com/LARG/jax-aht"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="hover:text-ink-300 transition-colors"
                        >
                            JaxAHT on GitHub
                        </a>
                        <a
                            href="/api/docs"
                            className="hover:text-ink-300 transition-colors"
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            API
                        </a>
                    </div>
                </div>
            </footer>
        </div>
    )
}
