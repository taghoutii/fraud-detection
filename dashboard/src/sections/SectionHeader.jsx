import { SECTIONS } from "../sections.js";

export default function SectionHeader({ id, title }) {
  const s = SECTIONS.find((x) => x.id === id);
  return (
    <header className="section-head">
      <div className="eyebrow">Step {s.step}</div>
      <h1 className="section-title">{title ?? s.label}</h1>
      <p className="section-question">{s.question}</p>
    </header>
  );
}

