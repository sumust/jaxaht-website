import { Route, Routes } from "react-router-dom"
import { Home } from "./pages/Home"
import { Play } from "./pages/Play"
import { Submit } from "./pages/Submit"
import { Leaderboard } from "./pages/Leaderboard"
import { Demo } from "./pages/Demo"
import { Prolific } from "./pages/Prolific"
import { Study } from "./pages/Study"
import { StudyComplete } from "./pages/StudyComplete"
import { About } from "./pages/About"
import { NotFound } from "./pages/NotFound"

export default function App() {
    return (
        <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/about" element={<About />} />
            <Route path="/:env/play" element={<Play />} />
            <Route path="/:env/submit" element={<Submit />} />
            <Route path="/:env/leaderboard" element={<Leaderboard />} />
            <Route path="/:env/demo" element={<Demo />} />
            <Route path="/:env/prolific" element={<Prolific />} />
            <Route path="/:env/study" element={<Study />} />
            <Route path="/:env/study/complete" element={<StudyComplete />} />
            <Route path="*" element={<NotFound />} />
        </Routes>
    )
}
