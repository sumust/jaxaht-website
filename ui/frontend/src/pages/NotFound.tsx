import { Link } from "react-router-dom"
import { Layout } from "../components/Layout"

export function NotFound() {
    return (
        <Layout title="404" subtitle="Page not found">
            <Link to="/" className="btn-primary">Back home</Link>
        </Layout>
    )
}
