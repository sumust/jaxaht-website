import { Layout } from "../components/Layout"

export function Diversity() {
    return (
        <Layout title="Diversity Viewer" subtitle="Held-out teammates on a 2D behavioral map, from the population-diversity features.">
            <iframe
                src="/diversity/index.html"
                title="Diversity Viewer"
                className="w-full rounded-lg border border-ink-800 bg-white"
                style={{ height: "calc(100vh - 200px)", minHeight: "600px" }}
            />
        </Layout>
    )
}
