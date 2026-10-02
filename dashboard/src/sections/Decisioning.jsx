import { useMemo, useState } from "react";
import {
  Area, AreaChart, CartesianGrid, ComposedChart, Line, LineChart, ReferenceArea, ReferenceDot,
  ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import SectionHeader from "./SectionHeader.jsx";
import { Card, Insight, LegendItem, Loading, Segmented, TooltipBox } from "../components/ui.jsx";
import { useData, fixed, num, pct } from "../lib/data.js";
import { confusionAt, decisionCurves, preparePredictions } from "../lib/curves.js";
import { axisProps, gridProps, mix, palette } from "../lib/theme.js";

const PCT_TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1];

export default function Decisioning() {
  const { data: report, error } = useData("results/final_classification_report.json");
  const { data: search } = useData("results/threshold_search.json");
  const { data: calibration } = useData("results/calibration.json");
  const { data: preds } = useData("predictions.json");

  const prep = useMemo(() => (preds ? preparePredictions(preds) : null), [preds]);
  const curves = useMemo(() => (prep ? decisionCurves(prep) : null), [prep]);
  const chosen = report?.threshold_selection.chosen_threshold;
  const [threshold, setThreshold] = useState(null);
  const t = threshold ?? chosen;

  if (!report || !search || !calibration) {
    return <section className="section"><SectionHeader id="decisioning" /><Loading height={420} error={error} /></section>;
  }

  return (
    <section className="section">
      <SectionHeader id="decisioning" />

      <div className="grid grid-2">
        <DriftFinding report={report} />
        <CalibrationFinding calibration={calibration} report={report} />
      </div>

      <Card
        className="mt"
        title="Threshold trade-off"
        sub={`Production rule: maximize precision while catching at least ${pct(report.threshold_rule.min_recall, 0)} of fraud. Chosen on validation data, applied unchanged to test.`}
        action={<ThresholdControl t={t} chosen={chosen} onChange={setThreshold} />}
      >
        <div className="threshold-layout">
          <ThresholdChart rows={search.thresholds} t={t} chosen={chosen} minRecall={report.threshold_rule.min_recall} onPick={setThreshold} />
          <Readout prep={prep} t={t} chosen={chosen} />
        </div>
      </Card>

      <div className="grid grid-2 mt">
        <Card title="Confusion matrix" sub={`All ${num(report.meta.n_test_rows)} test transactions at threshold ${t.toFixed(2)}`}>
          {prep ? <ConfusionMatrix prep={prep} t={t} /> : <Loading height={260} />}
        </Card>
        <CalibrationCard calibration={calibration} />
        <Card
          title="ROC curve"
          sub={curves ? `AUC ${fixed(curves.rocAuc)} · computed from every test prediction` : "Computing…"}
          action={<div className="legend"><LegendItem color={palette().navy} label={`At ${t.toFixed(2)}`} /></div>}
        >
          {curves ? <RocChart curves={curves} prep={prep} t={t} /> : <Loading height={280} />}
        </Card>
        <Card
          title="Precision–recall curve"
          sub={curves ? `Average precision ${fixed(curves.averagePrecision)} · no-skill line at the ${pct(prep.P / prep.n, 1)} fraud rate` : "Computing…"}
          action={<div className="legend"><LegendItem color={palette().navy} label={`At ${t.toFixed(2)}`} /></div>}
        >
          {curves ? <PrChart curves={curves} prep={prep} t={t} /> : <Loading height={280} />}
        </Card>
      </div>
    </section>
  );
}

// ---------- findings ----------

function DriftFinding({ report }) {
  const val = report.threshold_selection.validation_metrics_at_threshold;
  const test = report.test_metrics_at_chosen_threshold;
  return (
    <Insight
      icon="trend"
      eyebrow="Finding · monitor in production"
      title="Fraud patterns are shifting: precision fell while recall held"
      metrics={[
        { value: `${pct(val.precision, 0)} → ${pct(test.precision, 0)}`, label: "precision, validation → test" },
        { value: `${pct(val.recall, 0)} → ${pct(test.recall, 0)}`, label: "recall, validation → test" },
      ]}
    >
      The threshold kept its promise on unseen, later transactions: it still catches over{" "}
      {Math.floor(test.recall * 10) * 10}% of fraud. But a larger share of what it flags is now legitimate, the signature of
      drift between when the model learned and when it was tested. Alert precision is the early-warning metric to track.
    </Insight>
  );
}

function CalibrationFinding({ calibration, report }) {
  const bins = calibration.uniform_bins.filter((b) => b.count > 0);
  const firstOver = bins.find((b) => b.mean_predicted - b.observed_fraud_rate > 0.05);
  const high = bins.find((b) => b.bin_lower <= 0.85 && 0.85 < b.bin_upper);
  return (
    <Insight
      icon="gauge"
      eyebrow="Finding · read scores as ranks"
      title={`Scores above ~${firstOver.bin_lower.toFixed(1)} overstate the odds of fraud`}
      metrics={[
        { value: fixed(high.mean_predicted, 2), label: "average score in that band" },
        { value: pct(high.observed_fraud_rate, 0), label: "actually fraud" },
        { value: fixed(report.test_metrics_at_chosen_threshold.roc_auc), label: "ROC-AUC: ranking holds" },
      ]}
    >
      Transactions scored around {fixed(high.mean_predicted, 2)} turn out to be fraud {pct(high.observed_fraud_rate, 0)} of
      the time. The ordering is sound, so the higher score is still the riskier transaction, which is all the threshold
      needs. Wherever a literal probability matters, such as expected-loss estimates, the scores should be recalibrated first.
    </Insight>
  );
}

// ---------- threshold explorer ----------

function ThresholdControl({ t, chosen, onChange }) {
  return (
    <div className="threshold-control">
      <label className="threshold-label" htmlFor="threshold-slider">Threshold</label>
      <input
        id="threshold-slider"
        type="range"
        min={0.01}
        max={0.99}
        step={0.01}
        value={t}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <span className="threshold-value">{t.toFixed(2)}</span>
      <button className="reset" disabled={Math.abs(t - chosen) < 1e-9} onClick={() => onChange(null)}>
        Reset to {chosen}
      </button>
    </div>
  );
}

function ThresholdChart({ rows, t, chosen, minRecall, onPick }) {
  const p = palette();
  const recallOkUntil = Math.max(...rows.filter((r) => r.recall >= minRecall).map((r) => r.threshold));
  const pick = (state) => state?.activeLabel != null && onPick(Number(state.activeLabel));
  return (
    <div className="threshold-charts">
      <div style={{ height: 270 }}>
        <ResponsiveContainer>
          <LineChart data={rows} margin={{ top: 18, right: 12, left: -8, bottom: 0 }} onClick={pick} style={{ cursor: "pointer" }}>
            <CartesianGrid {...gridProps()} />
            <XAxis dataKey="threshold" type="number" domain={[0, 1]} ticks={PCT_TICKS} {...axisProps()} tickFormatter={(v) => v.toFixed(1)} />
            <YAxis domain={[0, 1]} ticks={PCT_TICKS} {...axisProps()} tickFormatter={(v) => pct(v, 0)} />
            <ReferenceArea x1={0} x2={recallOkUntil} fill={p.steel} fillOpacity={0.1} ifOverflow="hidden"
              label={{ value: `recall ≥ ${pct(minRecall, 0)}`, position: "insideTopLeft", fill: p.navy, fillOpacity: 0.6, fontSize: 11 }} />
            <ReferenceLine x={chosen} stroke={p.navy} strokeDasharray="4 4"
              label={{ value: `Production ${chosen}`, position: "top", fill: p.navy, fontSize: 11, fontWeight: 600 }} />
            {Math.abs(t - chosen) > 1e-9 && <ReferenceLine x={t} stroke={p.navy} strokeOpacity={0.6} />}
            <Tooltip content={<ThresholdTooltip />} cursor={{ stroke: p.navy, strokeOpacity: 0.2 }} />
            <Line dataKey="precision" stroke={p.navy} strokeWidth={2} dot={false} animationDuration={600} />
            <Line dataKey="recall" stroke={p.steel} strokeWidth={2} dot={false} animationDuration={600} />
            <Line dataKey="f1" stroke={p.steel} strokeWidth={2} strokeDasharray="5 4" strokeOpacity={0.7} dot={false} animationDuration={600} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="flag-strip-label">Transactions flagged</div>
      <div style={{ height: 82 }}>
        <ResponsiveContainer>
          <AreaChart data={rows} margin={{ top: 4, right: 12, left: -8, bottom: 0 }} onClick={pick} style={{ cursor: "pointer" }}>
            <XAxis dataKey="threshold" type="number" domain={[0, 1]} hide />
            <YAxis scale="log" domain={["auto", "auto"]} {...axisProps()} tickFormatter={(v) => (v >= 1000 ? `${Math.round(v / 1000)}k` : v)} width={52} ticks={[1000, 10000]} />
            <ReferenceLine x={chosen} stroke={p.navy} strokeDasharray="4 4" />
            {Math.abs(t - chosen) > 1e-9 && <ReferenceLine x={t} stroke={p.navy} strokeOpacity={0.6} />}
            <Tooltip content={<ThresholdTooltip />} cursor={{ stroke: p.navy, strokeOpacity: 0.2 }} />
            <Area dataKey="n_flagged" stroke={p.red} strokeWidth={1.5} fill={p.red} fillOpacity={0.12} animationDuration={600} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="legend legend-below">
        <LegendItem color={p.navy} label="Precision" />
        <LegendItem color={p.steel} label="Recall" />
        <LegendItem color={p.steel} label="F1" dashed />
        <LegendItem color={p.red} label="Flagged (log scale)" />
      </div>
    </div>
  );
}

function ThresholdTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload;
  const p = palette();
  return (
    <TooltipBox
      title={`Threshold ${r.threshold.toFixed(2)} · click to apply`}
      rows={[
        ["Precision", pct(r.precision), p.navy],
        ["Recall", pct(r.recall), p.steel],
        ["F1", fixed(r.f1)],
        ["Flagged", num(r.n_flagged), p.red],
      ]}
    />
  );
}

function Readout({ prep, t, chosen }) {
  if (!prep) return <Loading height={300} />;
  const c = confusionAt(prep, t);
  const base = confusionAt(prep, chosen);
  const delta = (a, b) => (Math.abs(t - chosen) < 1e-9 ? null : a - b);
  return (
    <div className="readout">
      <div className="readout-title">At threshold {t.toFixed(2)}</div>
      <ReadoutRow label="Fraud caught" value={pct(c.recall)} d={delta(c.recall, base.recall)} fmt={(d) => `${d >= 0 ? "+" : "−"}${(Math.abs(d) * 100).toFixed(1)} pts`} tone="risk" />
      <ReadoutRow label="Alerts that are fraud" value={pct(c.precision)} d={delta(c.precision, base.precision)} fmt={(d) => `${d >= 0 ? "+" : "−"}${(Math.abs(d) * 100).toFixed(1)} pts`} />
      <ReadoutRow label="F1" value={fixed(c.f1)} d={delta(c.f1, base.f1)} fmt={(d) => `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(3)}`} />
      <div className="readout-divider" />
      <ReadoutRow label="Flagged for review" value={num(c.flagged)} d={delta(c.flagged, base.flagged)} fmt={(d) => `${d >= 0 ? "+" : "−"}${num(Math.abs(d))}`} />
      <ReadoutRow label="False alarms" value={num(c.fp)} d={delta(c.fp, base.fp)} fmt={(d) => `${d >= 0 ? "+" : "−"}${num(Math.abs(d))}`} />
      <ReadoutRow label="Fraud missed" value={num(c.fn)} d={delta(c.fn, base.fn)} fmt={(d) => `${d >= 0 ? "+" : "−"}${num(Math.abs(d))}`} />
      <div className="readout-foot">Exact counts over every test transaction{Math.abs(t - chosen) < 1e-9 ? "" : ` · change vs ${chosen}`}</div>
    </div>
  );
}

function ReadoutRow({ label, value, d, fmt, tone }) {
  return (
    <div className="readout-row">
      <span className="readout-label">
        {tone === "risk" && <span className="chip-dot" style={{ background: "var(--risk-high)" }} />}
        {label}
      </span>
      <span className="readout-value">{value}</span>
      <span className="readout-delta">{d === null || d === 0 ? "" : fmt(d)}</span>
    </div>
  );
}

// ---------- confusion matrix ----------

function ConfusionMatrix({ prep, t }) {
  const c = confusionAt(prep, t);
  const fraud = c.tp + c.fn;
  const legit = c.fp + c.tn;
  const cells = [
    { key: "tp", label: "Fraud caught", value: c.tp, share: c.tp / fraud, of: "of fraud", cls: "cm-tp" },
    { key: "fn", label: "Fraud missed", value: c.fn, share: c.fn / fraud, of: "of fraud", cls: "cm-fn" },
    { key: "fp", label: "False alarms", value: c.fp, share: c.fp / legit, of: "of legitimate", cls: "cm-fp" },
    { key: "tn", label: "Cleared", value: c.tn, share: c.tn / legit, of: "of legitimate", cls: "cm-tn" },
  ];
  return (
    <div className="cm">
      <div />
      <div className="cm-col-head">Flagged</div>
      <div className="cm-col-head">Not flagged</div>
      <div className="cm-row-head">Actually fraud<span>{num(fraud)}</span></div>
      {cells.slice(0, 2).map((cell) => <CmCell key={cell.key} {...cell} />)}
      <div className="cm-row-head">Actually legitimate<span>{num(legit)}</span></div>
      {cells.slice(2).map((cell) => <CmCell key={cell.key} {...cell} />)}
    </div>
  );
}

function CmCell({ label, value, share, of, cls }) {
  return (
    <div className={`cm-cell ${cls}`} title={`${label}: ${num(value)} (${pct(share)} ${of})`}>
      <div className="cm-cell-label">{label}</div>
      <div className="cm-cell-value">{num(value)}</div>
      <div className="cm-cell-share">{pct(share)} {of}</div>
    </div>
  );
}

// ---------- ROC / PR ----------

function RocChart({ curves, prep, t }) {
  const p = palette();
  const c = confusionAt(prep, t);
  return (
    <div style={{ height: 290 }}>
      <ResponsiveContainer>
        <ComposedChart data={curves.points} margin={{ top: 10, right: 16, left: -8, bottom: 14 }}>
          <CartesianGrid {...gridProps()} vertical />
          <XAxis dataKey="fpr" type="number" domain={[0, 1]} ticks={PCT_TICKS} {...axisProps()} tickFormatter={(v) => pct(v, 0)}
            label={{ value: "False alarm rate (legitimate flagged)", position: "insideBottom", offset: -10, fill: p.navy, fillOpacity: 0.6, fontSize: 11 }} />
          <YAxis domain={[0, 1]} ticks={PCT_TICKS} {...axisProps()} tickFormatter={(v) => pct(v, 0)} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={p.navy} strokeOpacity={0.25} strokeDasharray="4 4" />
          <Tooltip content={<CurveTooltip kind="roc" />} cursor={{ stroke: p.navy, strokeOpacity: 0.2 }} />
          <Area dataKey="tpr" type="linear" stroke={p.steel} strokeWidth={2} fill={p.steel} fillOpacity={0.14} dot={false} isAnimationActive={false} />
          <ReferenceDot x={c.fpr} y={c.recall} r={6} fill={p.navy} stroke="#ffffff" strokeWidth={2} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function PrChart({ curves, prep, t }) {
  const p = palette();
  const c = confusionAt(prep, t);
  const data = curves.points.filter((pt) => pt.recall > 0);
  return (
    <div style={{ height: 290 }}>
      <ResponsiveContainer>
        <ComposedChart data={data} margin={{ top: 10, right: 16, left: -8, bottom: 14 }}>
          <CartesianGrid {...gridProps()} vertical />
          <XAxis dataKey="recall" type="number" domain={[0, 1]} ticks={PCT_TICKS} {...axisProps()} tickFormatter={(v) => pct(v, 0)}
            label={{ value: "Recall (fraud caught)", position: "insideBottom", offset: -10, fill: p.navy, fillOpacity: 0.6, fontSize: 11 }} />
          <YAxis domain={[0, 1]} ticks={PCT_TICKS} {...axisProps()} tickFormatter={(v) => pct(v, 0)} />
          <ReferenceLine y={prep.P / prep.n} stroke={p.navy} strokeOpacity={0.25} strokeDasharray="4 4" />
          <Tooltip content={<CurveTooltip kind="pr" />} cursor={{ stroke: p.navy, strokeOpacity: 0.2 }} />
          <Area dataKey="precision" type="linear" stroke={p.steel} strokeWidth={2} fill={p.steel} fillOpacity={0.14} dot={false} isAnimationActive={false} />
          <ReferenceDot x={c.recall} y={c.precision} r={6} fill={p.navy} stroke="#ffffff" strokeWidth={2} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}

function CurveTooltip({ active, payload, kind }) {
  if (!active || !payload?.length) return null;
  const pt = payload[0].payload;
  const rows = kind === "roc"
    ? [["Fraud caught", pct(pt.tpr)], ["False alarm rate", pct(pt.fpr, 2)]]
    : [["Recall", pct(pt.recall)], ["Precision", pct(pt.precision)]];
  return <TooltipBox title={`Threshold ${pt.threshold.toFixed(3)}`} rows={rows} />;
}

// ---------- calibration ----------

function CalibrationCard({ calibration }) {
  const [binning, setBinning] = useState("uniform_bins");
  const p = palette();
  const bins = calibration[`${binning}`].filter((b) => b.count > 0);
  return (
    <Card
      title="Calibration"
      sub={`Predicted score vs actual fraud rate · Brier score ${calibration.brier_score.toFixed(4)}`}
      action={<Segmented label="Binning" value={binning} onChange={setBinning}
        options={[{ value: "uniform_bins", label: "Even width" }, { value: "quantile_bins", label: "Even count" }]} />}
    >
      <div style={{ height: 290 }}>
        <ResponsiveContainer>
          <ComposedChart data={bins} margin={{ top: 10, right: 16, left: -8, bottom: 14 }}>
            <CartesianGrid {...gridProps()} vertical />
            <XAxis dataKey="mean_predicted" type="number" domain={[0, 1]} ticks={PCT_TICKS} {...axisProps()} tickFormatter={(v) => v.toFixed(1)}
              label={{ value: "Average predicted score", position: "insideBottom", offset: -10, fill: p.navy, fillOpacity: 0.6, fontSize: 11 }} />
            <YAxis domain={[0, 1]} ticks={PCT_TICKS} {...axisProps()} tickFormatter={(v) => pct(v, 0)} />
            <ReferenceArea x1={0.2} x2={1} y1={0} y2={1} fill={p.navy} fillOpacity={0.035}
              label={{ value: "below the diagonal = overconfident", position: "insideBottomRight", fill: p.navy, fillOpacity: 0.55, fontSize: 11 }} />
            <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={p.navy} strokeOpacity={0.3} strokeDasharray="4 4" />
            <Tooltip content={<CalibrationTooltip />} cursor={{ stroke: p.navy, strokeOpacity: 0.2 }} />
            <Line dataKey="observed_fraud_rate" stroke={p.steel} strokeWidth={2} isAnimationActive={false}
              dot={<BinDot maxCount={Math.max(...bins.map((b) => b.count))} />} activeDot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="legend legend-below">
        <LegendItem color={p.navy} label="Bin (size = transactions)" />
        <LegendItem color={mix(p.navy, "#ffffff", 0.6)} label="Perfect calibration" dashed />
      </div>
    </Card>
  );
}

// bin marker sized by how many transactions fall in the bin (area ~ count)
function BinDot({ cx, cy, payload, maxCount }) {
  if (cx == null || cy == null) return null;
  const r = 3.5 + 9 * Math.sqrt(payload.count / maxCount);
  const p = palette();
  return <circle cx={cx} cy={cy} r={r} fill={p.navy} fillOpacity={0.85} stroke="#ffffff" strokeWidth={1.5} />;
}

function CalibrationTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const b = payload[0].payload;
  return (
    <TooltipBox
      title={`Scores ${b.bin_lower.toFixed(2)}–${b.bin_upper.toFixed(2)}`}
      rows={[
        ["Average score", fixed(b.mean_predicted)],
        ["Actually fraud", pct(b.observed_fraud_rate)],
        ["Transactions", num(b.count)],
      ]}
    />
  );
}
