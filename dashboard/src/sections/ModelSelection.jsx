import {
  Bar, BarChart, CartesianGrid, ErrorBar, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import SectionHeader from "./SectionHeader.jsx";
import { Card, Insight, LegendItem, Loading, TooltipBox } from "../components/ui.jsx";
import { useData, fixed, signed } from "../lib/data.js";
import { axisProps, gridProps, mix, palette } from "../lib/theme.js";

const METRICS = [
  { key: "pr_auc", label: "PR-AUC" },
  { key: "roc_auc", label: "ROC-AUC" },
  { key: "f1", label: "F1" },
];
const WINNER = "lgbm";

export default function ModelSelection() {
  const { data: cv, error } = useData("results/cv_results.json");
  const { data: tuning } = useData("results/tuning_results.json");
  if (!cv || !tuning) return <section className="section"><SectionHeader id="model-selection" /><Loading height={420} error={error} /></section>;

  const models = Object.entries(cv.models)
    .map(([key, m]) => ({ key, ...m }))
    .sort((a, b) => b.metrics.pr_auc.mean - a.metrics.pr_auc.mean);
  const [first, second] = models;
  const cw = cv.models.logreg;
  const smote = cv.models.logreg_smote;
  const lift = tuning.tuned.pr_auc.mean - tuning.untuned.pr_auc.mean;

  return (
    <section className="section">
      <SectionHeader id="model-selection" />

      <Insight
        icon="trophy"
        eyebrow="Decision"
        title={`${first.label} runs in production`}
        metrics={[
          { value: fixed(first.metrics.pr_auc.mean), label: `PR-AUC, best of ${models.length}` },
          { value: signed(first.metrics.pr_auc.mean - second.metrics.pr_auc.mean), label: `ahead of ${second.label}` },
          { value: signed(lift), label: `PR-AUC from tuning → ${fixed(tuning.tuned.pr_auc.mean)}` },
          { value: `${(smote.cv_seconds / cw.cv_seconds).toFixed(1)}×`, label: "SMOTE's training cost, no meaningful gain" },
        ]}
      >
        {first.label} ranked first on PR-AUC across five candidates under time-ordered cross-validation,
        and was also the fastest to train. Class weighting handles the imbalance as well as SMOTE does
        ({fixed(cw.metrics.pr_auc.mean)} vs {fixed(smote.metrics.pr_auc.mean)} PR-AUC) without synthetic
        data, so every model uses it. Tuning then pushed {first.label} from {fixed(tuning.untuned.pr_auc.mean)} to{" "}
        {fixed(tuning.tuned.pr_auc.mean)}, improving on every fold.
      </Insight>

      <div className="grid grid-3" style={{ marginTop: 20 }}>
        <Card
          className="span-2"
          title="Five candidates, one winner"
          sub={`Mean over ${cv.meta.folds.length} time-ordered CV folds · bars show ±1 std · sorted by PR-AUC`}
          action={<div className="legend">{METRICS.map((m, i) => <LegendItem key={m.key} color={metricColor(i)} label={m.label} />)}</div>}
        >
          <CandidateChart models={models} />
        </Card>

        <Card title="Imbalance strategy" sub="Same logistic regression, two ways to handle 3.5% fraud">
          <ImbalanceCompare cw={cw} smote={smote} />
        </Card>

        <Card
          className="span-3"
          title={`Tuning ${tuning.label}`}
          sub={`RandomizedSearchCV, ${tuning.meta.n_iter} candidates, optimizing PR-AUC on the same time-ordered folds`}
          action={<div className="legend"><LegendItem color={palette().steel} label="Default settings" /><LegendItem color={palette().navy} label="Tuned" /></div>}
        >
          <TuningChart tuning={tuning} />
          <p className="footnote">
            Precision, recall and F1 here are at a 0.5 cut-off. Tuning made the model more selective at 0.5,
            and the production threshold is set separately (see Decisioning).
          </p>
        </Card>
      </div>
    </section>
  );
}

function metricColor(i) {
  const p = palette();
  return [p.navy, p.steel, mix(p.steel, "#ffffff", 0.5)][i];
}

function CandidateChart({ models }) {
  const data = models.map((m) => {
    const row = { name: m.label, key: m.key };
    for (const { key } of METRICS) {
      row[key] = m.metrics[key].mean;
      row[`${key}_std`] = m.metrics[key].std;
    }
    return row;
  });
  const p = palette();
  return (
    <div style={{ height: 340 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 16, right: 8, left: -8, bottom: 8 }} barGap={3} barCategoryGap="22%">
          <CartesianGrid {...gridProps()} />
          <XAxis dataKey="name" {...axisProps()} interval={0} tick={<ModelTick />} height={44} />
          <YAxis {...axisProps()} domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
          <Tooltip cursor={{ fill: p.navy, fillOpacity: 0.04 }} content={<CandidateTooltip />} />
          {METRICS.map((m, i) => (
            <Bar key={m.key} dataKey={m.key} name={m.label} fill={metricColor(i)} radius={[4, 4, 0, 0]} animationDuration={600}>
              <ErrorBar dataKey={`${m.key}_std`} width={4} stroke={p.navy} strokeOpacity={0.45} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function ModelTick({ x, y, payload }) {
  const p = palette();
  const winner = payload.value === "LightGBM";
  return (
    <g transform={`translate(${x},${y + 14})`}>
      <text textAnchor="middle" fill={p.navy} fontSize={12} fontWeight={winner ? 700 : 500} fillOpacity={winner ? 1 : 0.7}>
        {payload.value}
      </text>
      {winner && (
        <text y={16} textAnchor="middle" fill={p.steel} fontSize={10} fontWeight={700} letterSpacing="0.06em">
          SELECTED
        </text>
      )}
    </g>
  );
}

function CandidateTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <TooltipBox
      title={label}
      rows={METRICS.map((m, i) => [m.label, `${fixed(row[m.key])} ± ${fixed(row[`${m.key}_std`])}`, metricColor(i)])}
    />
  );
}

function ImbalanceCompare({ cw, smote }) {
  const rows = [
    { label: "PR-AUC", a: cw.metrics.pr_auc.mean, b: smote.metrics.pr_auc.mean, fmt: fixed },
    { label: "ROC-AUC", a: cw.metrics.roc_auc.mean, b: smote.metrics.roc_auc.mean, fmt: fixed },
    { label: "CV training time", a: cw.cv_seconds / 60, b: smote.cv_seconds / 60, fmt: (v) => `${v.toFixed(1)} min`, lowerIsBetter: true },
  ];
  return (
    <div className="versus">
      <div className="versus-head">
        <span className="versus-name versus-name-win"><span className="versus-key versus-key-strong" />class_weight<span className="adopted">Adopted</span></span>
        <span className="versus-name"><span className="versus-key" />SMOTE</span>
      </div>
      {rows.map((r) => {
        const max = Math.max(r.a, r.b);
        return (
          <div key={r.label} className="versus-row">
            <div className="versus-label">{r.label}</div>
            <div className="versus-bars">
              <VersusBar value={r.a} max={max} text={r.fmt(r.a)} strong />
              <VersusBar value={r.b} max={max} text={r.fmt(r.b)} />
            </div>
          </div>
        );
      })}
      <p className="versus-note">
        Accuracy is a statistical tie ({signed(smote.metrics.pr_auc.mean - cw.metrics.pr_auc.mean)} PR-AUC,
        well inside the ±{fixed(cw.metrics.pr_auc.std, 2)} fold spread). Class weighting gets there with
        no synthetic rows and half the compute.
      </p>
    </div>
  );
}

function VersusBar({ value, max, text, strong }) {
  return (
    <div className="versus-bar-row">
      <div className="versus-bar-track">
        <div className={`versus-bar ${strong ? "versus-bar-strong" : ""}`} style={{ width: `${(value / max) * 100}%` }} />
      </div>
      <span className={`versus-bar-text ${strong ? "versus-bar-text-strong" : ""}`}>{text}</span>
    </div>
  );
}

function TuningChart({ tuning }) {
  const keys = [
    ["pr_auc", "PR-AUC"], ["roc_auc", "ROC-AUC"], ["f1", "F1"], ["precision", "Precision"], ["recall", "Recall"],
  ];
  const data = keys.map(([k, label]) => ({
    name: label,
    untuned: tuning.untuned[k].mean,
    tuned: tuning.tuned[k].mean,
    untuned_std: tuning.untuned[k].std,
    tuned_std: tuning.tuned[k].std,
    delta: tuning.tuned[k].mean - tuning.untuned[k].mean,
  }));
  const p = palette();
  return (
    <div style={{ height: 300 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 28, right: 8, left: -8, bottom: 0 }} barGap={4} barCategoryGap="28%">
          <CartesianGrid {...gridProps()} />
          <XAxis dataKey="name" {...axisProps()} />
          <YAxis {...axisProps()} domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
          <Tooltip
            cursor={{ fill: p.navy, fillOpacity: 0.04 }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipBox
                  title={label}
                  rows={[
                    ["Default", `${fixed(payload[0].payload.untuned)} ± ${fixed(payload[0].payload.untuned_std)}`, p.steel],
                    ["Tuned", `${fixed(payload[0].payload.tuned)} ± ${fixed(payload[0].payload.tuned_std)}`, p.navy],
                    ["Change", signed(payload[0].payload.delta)],
                  ]}
                />
              ) : null
            }
          />
          <Bar dataKey="untuned" fill={p.steel} radius={[4, 4, 0, 0]} animationDuration={600} />
          <Bar dataKey="tuned" fill={p.navy} radius={[4, 4, 0, 0]} animationDuration={600}>
            <LabelList dataKey="delta" position="top" formatter={(v) => signed(v)} fill={p.navy} fontSize={12} fontWeight={600} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
