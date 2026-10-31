import {
  Bar, BarChart, CartesianGrid, Cell, ErrorBar, ResponsiveContainer, Tooltip, XAxis, YAxis,
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
// steps of 0.2 so every tick label is exact at one decimal
const TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1];
const COUNT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];
const countWord = (n) => COUNT_WORDS[n] ?? String(n);

export default function ModelSelection() {
  const { data: cv, error } = useData("results/cv_results.json");
  const { data: tuning } = useData("results/tuning_results.json");
  if (!cv || !tuning) return <section className="section"><SectionHeader id="model-selection" /><Loading height={420} error={error} /></section>;

  const models = Object.entries(cv.models)
    .map(([key, m]) => ({ key, ...m }))
    .sort((a, b) => b.metrics.pr_auc.mean - a.metrics.pr_auc.mean);
  const [first] = models;
  const fastest = models.every((m) => m.cv_seconds >= first.cv_seconds);
  const cw = cv.models.logreg.metrics.pr_auc;
  const smote = cv.models.logreg_smote.metrics.pr_auc;
  const smoteGain = smote.mean - cw.mean;
  const lift = tuning.tuned.pr_auc.mean - tuning.untuned.pr_auc.mean;
  const title = (s) => s[0].toUpperCase() + s.slice(1);

  return (
    <section className="section">
      <SectionHeader id="model-selection" />

      <Insight
        icon="trophy"
        eyebrow="Decision"
        title={`${first.label} is the production model`}
        metrics={[
          { value: fixed(first.metrics.pr_auc.mean), label: `PR-AUC, best of ${models.length}` },
          { value: signed(lift), label: "from tuning" },
          // a gain inside the fold-to-fold spread is no gain
          { value: Math.abs(smoteGain) < cw.std ? "No gain" : signed(smoteGain), label: "from SMOTE" },
        ]}
      >
        {first.label} ranked first on PR-AUC across {countWord(models.length)} candidates{fastest ? " and trained fastest" : ""}.
        Class weighting matched SMOTE with no synthetic data, and tuning lifted PR-AUC to {fixed(tuning.tuned.pr_auc.mean)}.
      </Insight>

      <div className="grid grid-3" style={{ marginTop: 20 }}>
        <Card
          className="span-2"
          title={`${title(countWord(models.length))} candidates, one winner`}
          action={<div className="legend">{METRICS.map((m, i) => <LegendItem key={m.key} color={metricColor(i)} label={m.label} />)}</div>}
        >
          <CandidateChart models={models} />
        </Card>

        <Card title={`PR-AUC before and after tuning ${tuning.label}`}>
          <TuningChart tuning={tuning} />
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
    <div style={{ height: 320 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 16, right: 8, left: -8, bottom: 8 }} barGap={3} barCategoryGap="22%">
          <CartesianGrid {...gridProps()} />
          <XAxis dataKey="name" {...axisProps()} interval={0} tick={<ModelTick />} height={44} />
          <YAxis {...axisProps()} domain={[0, 1]} ticks={TICKS} tickFormatter={(v) => v.toFixed(1)} />
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

function TuningChart({ tuning }) {
  const p = palette();
  const data = [
    { name: "Default", value: tuning.untuned.pr_auc.mean, std: tuning.untuned.pr_auc.std, color: p.steel },
    { name: "Tuned", value: tuning.tuned.pr_auc.mean, std: tuning.tuned.pr_auc.std, color: p.navy },
  ];
  return (
    <div style={{ height: 320 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 16, right: 8, left: -8, bottom: 8 }} barCategoryGap="30%">
          <CartesianGrid {...gridProps()} />
          <XAxis dataKey="name" {...axisProps()} height={44} />
          <YAxis {...axisProps()} domain={[0, 1]} ticks={TICKS} tickFormatter={(v) => v.toFixed(1)} />
          <Tooltip
            cursor={{ fill: p.navy, fillOpacity: 0.04 }}
            content={({ active, payload, label }) =>
              active && payload?.length ? (
                <TooltipBox title={label} rows={[["PR-AUC", `${fixed(payload[0].payload.value)} ± ${fixed(payload[0].payload.std)}`, payload[0].payload.color]]} />
              ) : null
            }
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]} animationDuration={600}>
            {data.map((d) => <Cell key={d.name} fill={d.color} />)}
            <ErrorBar dataKey="std" width={4} stroke={p.navy} strokeOpacity={0.45} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
