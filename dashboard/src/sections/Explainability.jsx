import { useEffect, useMemo, useState } from "react";
import {
  Bar, BarChart, Cell, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import SectionHeader from "./SectionHeader.jsx";
import Beeswarm from "../components/Beeswarm.jsx";
import { Card, LegendItem, Loading, Segmented, TooltipBox } from "../components/ui.jsx";
import { fixed, formatScore, formatValue, num, sigmoid, useData } from "../lib/data.js";
import { axisProps, gridProps, mix, palette } from "../lib/theme.js";

const WATERFALL_TOP = 8;

export default function Explainability() {
  const [view, setView] = useState("global");
  const [txId, setTxId] = useState(null);
  const { data: sample, error } = useData("transactions.json");

  const openTransaction = (id) => {
    setTxId(id);
    setView("local");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <section className="section">
      <div className="section-head-row">
        <SectionHeader id="explainability" />
        <Segmented
          label="Explanation scope"
          value={view}
          onChange={setView}
          options={[{ value: "global", label: "Global · all transactions" }, { value: "local", label: "Local · one transaction" }]}
        />
      </div>
      {!sample ? <Loading height={420} error={error} />
        : view === "global"
          ? <GlobalView sample={sample} onSelect={openTransaction} />
          : <LocalView sample={sample} txId={txId} setTxId={setTxId} />}
    </section>
  );
}

// ---------- global ----------

function GlobalView({ sample, onSelect }) {
  const { data: global } = useData("shap_global.json");
  const p = palette();
  return (
    <div className="fade-in">
      <Card
        title="What drives fraud scores"
        sub="Each dot is a test transaction. Click one to explain it."
        action={
          <div className="legend">
            <span className="gradient-legend">
              <span>Low</span>
              <span className="gradient-bar" style={{ background: `linear-gradient(90deg, ${p.steel}, ${p.navy})` }} />
              <span>High feature value</span>
            </span>
            <LegendItem color={mix(p.navy, "#ffffff", 0.78)} label="Missing" />
          </div>
        }
      >
        {global ? <Beeswarm features={global.features} transactions={sample.transactions} onSelect={onSelect} /> : <Loading height={640} />}
      </Card>
    </div>
  );
}

// ---------- local ----------

const OUTCOMES = {
  all: { label: "All", test: () => true },
  caught: { label: "Fraud caught", test: (t) => t.is_fraud && t.flagged },
  missed: { label: "Fraud missed", test: (t) => t.is_fraud && !t.flagged },
  false_alarm: { label: "False alarms", test: (t) => !t.is_fraud && t.flagged },
  cleared: { label: "Cleared", test: (t) => !t.is_fraud && !t.flagged },
};

function outcomeOf(t) {
  return Object.keys(OUTCOMES).slice(1).find((k) => OUTCOMES[k].test(t));
}

function LocalView({ sample, txId, setTxId }) {
  const [filter, setFilter] = useState("all");
  const [query, setQuery] = useState("");
  const byScore = useMemo(() => [...sample.transactions].sort((a, b) => b.proba - a.proba), [sample]);

  // default: the highest-scored fraud the model caught
  useEffect(() => {
    if (txId === null) setTxId(byScore.find((t) => t.is_fraud && t.flagged).id);
  }, [txId, byScore, setTxId]);

  const counts = Object.fromEntries(Object.entries(OUTCOMES).map(([k, o]) => [k, byScore.filter(o.test).length]));
  const matches = byScore.filter((t) => OUTCOMES[filter].test(t) && String(t.id).includes(query.trim()));
  const selected = sample.transactions.find((t) => t.id === txId);

  return (
    <div className="local fade-in">
      <div className="card picker">
        <input className="search" type="search" placeholder="Search transaction ID" value={query} onChange={(e) => setQuery(e.target.value)} />
        <div className="filter-chips">
          {Object.entries(OUTCOMES).map(([k, o]) => (
            <button key={k} className={`filter-chip filter-${k}`} aria-pressed={filter === k} onClick={() => setFilter(k)}>
              {o.label}<span>{counts[k]}</span>
            </button>
          ))}
        </div>
        <div className="picker-list" role="listbox" aria-label="Transactions">
          {matches.slice(0, 150).map((t) => (
            <button key={t.id} role="option" aria-selected={t.id === txId} className="picker-row" onClick={() => setTxId(t.id)}>
              <span className={`outcome-dot outcome-${outcomeOf(t)}`} />
              <span className="picker-id">{t.id}</span>
              <span className="picker-bar"><span style={{ width: `${Math.max(t.proba * 100, 1.5)}%` }} className={t.flagged ? "picker-bar-flagged" : ""} /></span>
              <span className="picker-score">{formatScore(t.proba)}</span>
            </button>
          ))}
          {matches.length > 150 && <div className="picker-more">{num(matches.length - 150)} more · refine the search</div>}
          {matches.length === 0 && <div className="picker-more">No transactions match</div>}
        </div>
      </div>

      {selected ? <TransactionDetail tx={selected} base={sample.base_value} threshold={sample.chosen_threshold} /> : <Loading height={520} />}
    </div>
  );
}

function TransactionDetail({ tx, base, threshold }) {
  const { data: chunk } = useData(`shap_local/${String(tx.chunk).padStart(2, "0")}.json`);
  const row = chunk?.rows[String(tx.id)];

  const contribs = useMemo(() => {
    if (!row) return null;
    return chunk.features
      .map((feature, i) => ({ feature, shap: row.shap[i], value: row.values[i] }))
      .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
  }, [chunk, row]);

  return (
    <div className="card detail" key={tx.id}>
      <div className="detail-head">
        <div>
          <div className="eyebrow">Transaction</div>
          <div className="detail-id">{tx.id}</div>
        </div>
        <div className="detail-badges">
          <span className={`chip ${tx.flagged ? "chip-risk" : "chip-safe"}`}>
            <span className="chip-dot" style={{ background: tx.flagged ? "var(--risk-alert)" : "var(--safe)" }} />
            {tx.flagged ? "Flagged" : "Not flagged"}
          </span>
          <span className={`chip ${tx.is_fraud ? "chip-risk" : "chip-safe"}`}>
            <span className="chip-dot" style={{ background: tx.is_fraud ? "var(--risk-high)" : "var(--safe)" }} />
            {tx.is_fraud ? "Confirmed fraud" : "Legitimate"}
          </span>
        </div>
      </div>

      <div className="detail-stats">
        <ScoreGauge score={tx.proba} threshold={threshold} />
        <div className="detail-stat"><div className="stat-label">Amount</div><div className="detail-stat-value">{tx.amount === null ? "—" : `$${formatValue(tx.amount)}`}</div></div>
      </div>

      {contribs ? (
        <>
          <Waterfall contribs={contribs} base={base} threshold={threshold} />
        </>
      ) : <Loading height={460} />}
    </div>
  );
}

function ScoreGauge({ score, threshold }) {
  return (
    <div className="gauge">
      <div className="stat-label">Fraud score</div>
      <div className={`gauge-value ${score >= threshold ? "gauge-value-risk" : ""}`}>{formatScore(score)}</div>
      <div className="gauge-track">
        <div className={`gauge-fill ${score >= threshold ? "gauge-fill-risk" : ""}`} style={{ width: `${score * 100}%` }} />
        <div className="gauge-threshold" style={{ left: `${threshold * 100}%` }} title={`Threshold ${threshold}`} />
      </div>
    </div>
  );
}

function Waterfall({ contribs, base, threshold }) {
  const p = palette();
  const top = contribs.slice(0, WATERFALL_TOP);
  const rest = contribs.slice(WATERFALL_TOP);
  const restSum = rest.reduce((s, c) => s + c.shap, 0);

  let cum = base;
  const rows = top.map((c) => {
    const start = cum;
    cum += c.shap;
    return { name: `${c.feature} = ${formatValue(c.value)}`, range: [start, cum], ...c, start, end: cum };
  });
  rows.push({ name: `${rest.length} other features`, range: [cum, cum + restSum], shap: restSum, start: cum, end: cum + restSum, other: true });
  const final = cum + restSum;
  const thresholdLogit = Math.log(threshold / (1 - threshold));

  const ends = rows.flatMap((r) => r.range).concat([base, final, thresholdLogit]);
  // round axis to a 1/2/5 step so ticks land on readable log-odds values
  const span = Math.max(...ends) - Math.min(...ends) + 1;
  const step = [1, 2, 5, 10].find((s) => span / s <= 8) ?? 20;
  const lo = Math.floor((Math.min(...ends) - 0.5) / step) * step;
  const hi = Math.ceil((Math.max(...ends) + 0.5) / step) * step;
  const ticks = Array.from({ length: Math.round((hi - lo) / step) + 1 }, (_, i) => lo + i * step);

  return (
    <div className="waterfall">
      <div className="waterfall-head">
        <div>
          <div className="card-title">How the score was built</div>
          <div className="card-sub">From a typical transaction to this one.</div>
        </div>
        <div className="legend">
          <LegendItem color={p.red} label="Pushes toward fraud" />
          <LegendItem color={p.steel} label="Pushes toward legitimate" />
          <LegendItem color={p.navy} label="Typical transaction" dashed />
          <LegendItem color={p.navy} label="This transaction" />
          <LegendItem color={p.red} label={`Flag line (${threshold})`} />
        </div>
      </div>
      <div style={{ height: rows.length * 32 + 64 }}>
        <ResponsiveContainer>
          <BarChart data={rows} layout="vertical" margin={{ top: 8, right: 24, left: 8, bottom: 8 }} barCategoryGap="24%">
            <CartesianGrid {...gridProps()} horizontal={false} vertical />
            <XAxis type="number" domain={[lo, hi]} ticks={ticks} {...axisProps()} tickFormatter={(v) => v.toFixed(0)} />
            <YAxis type="category" dataKey="name" {...axisProps()} width={230} interval={0} tick={<FeatureTick />} />
            <ReferenceLine x={base} stroke={p.navy} strokeOpacity={0.45} strokeDasharray="3 3" />
            <ReferenceLine x={thresholdLogit} stroke={p.red} strokeOpacity={0.7} />
            <ReferenceLine x={final} stroke={p.navy} strokeWidth={2} />
            <Tooltip cursor={{ fill: p.navy, fillOpacity: 0.04 }} content={<WaterfallTooltip />} />
            <Bar dataKey="range" radius={3} animationDuration={500}>
              {rows.map((r) => (
                <Cell key={r.name} fill={r.shap >= 0 ? p.red : p.steel} fillOpacity={r.other ? 0.45 : 0.9} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function FeatureTick({ x, y, payload }) {
  const p = palette();
  const [feature, value] = payload.value.split(" = ");
  return (
    <text x={x} y={y} dy={4} textAnchor="end" fontSize={12} fill={p.navy}>
      <tspan fontWeight={600}>{feature}</tspan>
      {value !== undefined && <tspan fillOpacity={0.6}> = {value}</tspan>}
    </text>
  );
}

function WaterfallTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload;
  const p = palette();
  return (
    <TooltipBox
      title={r.other ? r.name : r.feature}
      rows={[
        ...(r.other ? [] : [["Value", formatValue(r.value)]]),
        ["SHAP (log-odds)", `${r.shap >= 0 ? "+" : "−"}${fixed(Math.abs(r.shap))}`, r.shap >= 0 ? p.red : p.steel],
        ["Score", `${formatScore(sigmoid(r.start))} → ${formatScore(sigmoid(r.end))}`],
      ]}
    />
  );
}
