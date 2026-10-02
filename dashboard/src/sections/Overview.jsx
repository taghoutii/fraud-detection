import SectionHeader, { Slot } from "./SectionHeader.jsx";

export default function Overview() {
  return (
    <section className="section">
      <SectionHeader id="overview" title="Fraud Risk Console" />
      <div className="grid grid-3">
        <div className="span-2">
          <Slot title="Production model" sub="Chosen model + headline test metric" note="LightGBM · PR-AUC 0.578 badge" height={180} />
        </div>
        <Slot title="Mission" note="One-line mission statement" height={180} />
        <Slot title="Transactions" note="Dataset size chip" height={70} />
        <Slot title="Fraud rate" note="Fraud rate chip" height={70} />
        <Slot title="Test window" note="Test-set time span chip" height={70} />
      </div>
      <div className="legend-demo">
        <span className="chip chip-safe"><span className="chip-dot" style={{ background: "var(--safe)" }} />Legitimate</span>
        <span className="chip chip-risk"><span className="chip-dot" style={{ background: "var(--risk-alert)" }} />Flagged</span>
        <span className="chip chip-risk"><span className="chip-dot" style={{ background: "var(--risk-high)" }} />Confirmed fraud</span>
      </div>
    </section>
  );
}
