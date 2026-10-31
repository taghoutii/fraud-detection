import { Loading } from "../components/ui.jsx";
import { useData, num, pct, fixed } from "../lib/data.js";

const MISSION = "Fraud scoring for card transactions";

export default function Overview() {
  const { data: ov, error } = useData("overview.json");
  const { data: report } = useData("results/final_classification_report.json");

  if (!ov || !report) return <section className="section"><Loading height={420} error={error} /></section>;
  const atThreshold = report.test_metrics_at_chosen_threshold;
  const cm = atThreshold.confusion_matrix;

  return (
    <section className="section">
      <header className="overview-head">
        <h1 className="mission">{MISSION}</h1>
      </header>

      <div className="hero">
        <div className="hero-main">
          <div className="hero-model">
            <span className="live-pill"><span className="status-dot" />In production</span>
            <h1 className="hero-model-name">{ov.model}</h1>
            <p className="hero-model-sub">Gradient-boosted trees</p>
          </div>
          <div className="hero-metric">
            <div className="hero-metric-value">{fixed(ov.test_pr_auc)}</div>
            <div className="hero-metric-label">PR-AUC on held-out test</div>
            <div className="hero-metric-hint">
              {(ov.test_pr_auc / ov.test_fraud_rate).toFixed(0)}× the no-skill baseline · ROC-AUC {fixed(ov.test_roc_auc)}
            </div>
          </div>
        </div>

        <div className="hero-decisions">
          <div className="hero-decisions-title">At the {ov.chosen_threshold} decision threshold</div>
          <div className="hero-decision-grid">
            <HeroFigure value={pct(atThreshold.recall, 0)} label="of fraud caught" tone="risk" />
            <HeroFigure value={pct(atThreshold.precision, 0)} label="of alerts are fraud" />
            <HeroFigure value={pct(atThreshold.flag_rate)} label="of transactions flagged" />
          </div>
          <div className="hero-decisions-foot">
            {num(cm.tp)} frauds caught · {num(cm.fp)} false alarms · out of {num(ov.n_test)} test transactions
          </div>
        </div>
      </div>

      <div className="chips-row">
        <StatChip label="Transactions scored" value={num(ov.n_transactions)} />
        <StatChip label="Fraud rate" value={pct(ov.test_fraud_rate, 2)} tone="risk" />
        <StatChip label="Test window" value={`${ov.test_span_days.toFixed(0)} days`} />
      </div>
    </section>
  );
}

function HeroFigure({ value, label, tone }) {
  return (
    <div className={`hero-figure ${tone === "risk" ? "hero-figure-risk" : ""}`}>
      <div className="hero-figure-value">{value}</div>
      <div className="hero-figure-label">{label}</div>
    </div>
  );
}

function StatChip({ label, value, tone }) {
  return (
    <div className="stat-chip">
      <div className="stat-chip-label">
        {tone === "risk" && <span className="chip-dot" style={{ background: "var(--risk-alert)" }} />}
        {label}
      </div>
      <div className="stat-chip-value">{value}</div>
    </div>
  );
}
