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

// Shell-review placeholder: a titled card with a dashed slot where the real content will go.
export function Slot({ title, sub, note, height }) {
  return (
    <div className="card">
      <h2 className="card-title">{title}</h2>
      {sub && <p className="card-sub">{sub}</p>}
      <div className="slot" style={height ? { "--slot-h": `${height}px` } : undefined}>{note}</div>
    </div>
  );
}
