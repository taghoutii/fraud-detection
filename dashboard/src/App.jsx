import { useEffect, useState } from "react";
import { SECTIONS } from "./sections.js";
import Overview from "./sections/Overview.jsx";
import ModelSelection from "./sections/ModelSelection.jsx";
import Decisioning from "./sections/Decisioning.jsx";
import Explainability from "./sections/Explainability.jsx";

const VIEWS = {
  overview: Overview,
  "model-selection": ModelSelection,
  decisioning: Decisioning,
  explainability: Explainability,
};

function currentSection() {
  const id = window.location.hash.replace("#", "");
  return VIEWS[id] ? id : "overview";
}

export default function App() {
  const [active, setActive] = useState(currentSection);

  useEffect(() => {
    const onHash = () => setActive(currentSection());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = (id) => {
    window.location.hash = id;
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const View = VIEWS[active];
  return (
    <div className="shell">
      <nav className="nav" aria-label="Sections">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6l7-3z" />
            </svg>
          </div>
          <div>
            <div className="brand-name">Fraud Risk Console</div>
            <div className="brand-sub">Card transactions · IEEE-CIS</div>
          </div>
        </div>

        <ul className="nav-list">
          {SECTIONS.map((s) => (
            <li key={s.id}>
              <button className="nav-item" aria-current={active === s.id ? "page" : undefined} onClick={() => go(s.id)}>
                <span className="nav-step">{s.step}</span>
                <span>
                  <span className="nav-label">{s.label}</span>
                  <span className="nav-question">{s.question}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>

        <div className="nav-footer">
          <div><span className="status-dot" />Model serving · LightGBM</div>
          <div>Decision threshold 0.13</div>
        </div>
      </nav>

      <main className="main">
        {/* key remounts the view so each section plays its entrance transition */}
        <View key={active} />
      </main>
    </div>
  );
}
