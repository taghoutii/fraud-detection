import SectionHeader, { Slot } from "./SectionHeader.jsx";

export default function ModelSelection() {
  return (
    <section className="section">
      <SectionHeader id="model-selection" />
      <div className="grid grid-3">
        <div className="span-3">
          <Slot title="Conclusion" note="Callout: LightGBM won · class_weight over SMOTE · tuning lift" height={90} />
        </div>
        <div className="span-2">
          <Slot title="Five candidates, time-aware CV" sub="PR-AUC · ROC-AUC · F1" note="Grouped bar chart (cv_results.json)" height={340} />
        </div>
        <Slot title="Tuning LightGBM" sub="Untuned vs tuned" note="Before/after chart (tuning_results.json)" height={340} />
      </div>
    </section>
  );
}
