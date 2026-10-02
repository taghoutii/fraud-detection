import SectionHeader, { Slot } from "./SectionHeader.jsx";

export default function Decisioning() {
  return (
    <section className="section">
      <SectionHeader id="decisioning" />
      <div className="grid grid-2">
        <Slot title="Finding: precision drift" note="Validation → test precision 0.67 → 0.45, recall held" height={90} />
        <Slot title="Finding: overconfidence above ~0.2" note="Scored ~0.85 → ~49% actually fraud" height={90} />
        <div className="span-2">
          <Slot title="Threshold trade-off" sub="Rule: maximize precision subject to recall ≥ 60%"
                note="Interactive curve + slider, 0.13 marked (threshold_search.json)" height={320} />
        </div>
        <Slot title="Confusion matrix" note="Computed client-side from predictions.json" height={260} />
        <Slot title="Calibration" note="Reliability curve (calibration.json)" height={260} />
        <Slot title="ROC curve" note="Computed client-side" height={260} />
        <Slot title="Precision–recall curve" note="Computed client-side" height={260} />
      </div>
    </section>
  );
}
