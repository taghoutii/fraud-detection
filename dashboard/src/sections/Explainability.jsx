import SectionHeader, { Slot } from "./SectionHeader.jsx";

export default function Explainability() {
  return (
    <section className="section">
      <SectionHeader id="explainability" />
      <div className="grid grid-2">
        <div className="span-2">
          <Slot title="Global · SHAP summary" sub="Top 20 features across the 2,000-transaction sample"
                note="Beeswarm (shap_global.json)" height={380} />
        </div>
        <Slot title="LightGBM native importance" note="Ranked by gain" height={300} />
        <Slot title="Mean |SHAP| importance" note="Ranked by mean |SHAP|" height={300} />
        <div className="span-2">
          <Slot title="Local · one transaction" sub="Search the sample, see what pushed its score up or down"
                note="Picker + waterfall (transactions.json + shap_local/)" height={380} />
        </div>
      </div>
    </section>
  );
}
