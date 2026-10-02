import { useEffect, useMemo, useRef, useState } from "react";
import { palette, mix } from "../lib/theme.js";
import { fixed } from "../lib/data.js";

// SHAP summary ("beeswarm"): one row per feature, one dot per sampled transaction.
// x = SHAP value (log-odds; right = pushes toward fraud), color = the feature's value
// (steel = low, navy = high, gray = missing). Drawn on canvas: 20 x 2,000 points is too many
// SVG nodes to stay smooth. Hover shows the dot; click opens that transaction.
const ROW_H = 30;
const LABEL_W = 190;
const PAD = { top: 8, right: 20, bottom: 36 };

export default function Beeswarm({ features, transactions, onSelect }) {
  const wrapRef = useRef(null);
  const canvasRef = useRef(null);
  const [width, setWidth] = useState(900);
  const [hover, setHover] = useState(null);
  const height = PAD.top + features.length * ROW_H + PAD.bottom;

  useEffect(() => {
    const ro = new ResizeObserver(([entry]) => setWidth(Math.max(480, entry.contentRect.width)));
    ro.observe(wrapRef.current);
    return () => ro.disconnect();
  }, []);

  // x domain: clip the extreme 0.5% so a few outliers don't squash everything else
  const domain = useMemo(() => {
    const all = features.flatMap((f) => f.shap).sort((a, b) => a - b);
    const q = (p) => all[Math.floor(p * (all.length - 1))];
    const lo = Math.min(q(0.002), -0.5);
    const hi = Math.max(q(0.998), 0.5);
    return [lo, hi];
  }, [features]);

  // dot positions, computed once per layout: beeswarm-style vertical offsets by local density
  const dots = useMemo(() => {
    const plotW = width - LABEL_W - PAD.right;
    const x = (v) => LABEL_W + ((Math.min(Math.max(v, domain[0]), domain[1]) - domain[0]) / (domain[1] - domain[0])) * plotW;
    const out = [];
    features.forEach((f, row) => {
      const cy = PAD.top + row * ROW_H + ROW_H / 2;
      const buckets = new Map();
      f.shap.forEach((s, i) => {
        const px = x(s);
        const b = Math.round(px / 2.5);
        const k = buckets.get(b) ?? 0;
        buckets.set(b, k + 1);
        const offset = ((k % 2 ? 1 : -1) * Math.ceil(k / 2) * 1.6) % (ROW_H / 2 - 3);
        out.push({ x: px, y: cy + offset, row, i, shap: s, pct: f.value_pct[i] });
      });
    });
    //drawn missing first (underneath), then by value so high values sit on top; sorted once, not per hover
    return out.sort((a, b) => (a.pct ?? -1) - (b.pct ?? -1));
  }, [features, width, domain]);

  useEffect(() => {
    const p = palette();
    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const plotW = width - LABEL_W - PAD.right;
    const x = (v) => LABEL_W + ((v - domain[0]) / (domain[1] - domain[0])) * plotW;

    // row bands + labels
    ctx.font = "500 12px Inter Variable, system-ui, sans-serif";
    ctx.textBaseline = "middle";
    features.forEach((f, row) => {
      const cy = PAD.top + row * ROW_H + ROW_H / 2;
      if (row % 2 === 0) {
        ctx.fillStyle = mix(p.navy, "#ffffff", 0.97);
        ctx.fillRect(LABEL_W, cy - ROW_H / 2, plotW, ROW_H);
      }
      ctx.fillStyle = p.navy;
      ctx.textAlign = "right";
      ctx.fillText(f.feature, LABEL_W - 14, cy);
    });

    // zero line + axis
    ctx.strokeStyle = mix(p.navy, "#ffffff", 0.7);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x(0), PAD.top);
    ctx.lineTo(x(0), height - PAD.bottom);
    ctx.stroke();
    ctx.fillStyle = mix(p.navy, "#ffffff", 0.35);
    ctx.textAlign = "center";
    const step = niceStep(domain[1] - domain[0]);
    for (let v = Math.ceil(domain[0] / step) * step; v <= domain[1] + 1e-9; v += step) {
      ctx.fillText(v.toFixed(step < 1 ? 1 : 0), x(v), height - PAD.bottom + 14);
    }
    ctx.fillText("← lowers fraud score      SHAP value (log-odds)      raises fraud score →", LABEL_W + plotW / 2, height - 8);

    const missing = mix(p.navy, "#ffffff", 0.78);
    for (const d of dots) {
      ctx.fillStyle = d.pct === null ? missing : mix(p.steel, p.navy, d.pct);
      ctx.globalAlpha = d.pct === null ? 0.7 : 0.85;
      ctx.beginPath();
      ctx.arc(d.x, d.y, 2.3, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (hover) {
      ctx.strokeStyle = p.navy;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(hover.x, hover.y, 5.5, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [dots, features, width, height, domain, hover]);

  const findDot = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const row = Math.floor((my - PAD.top) / ROW_H);
    if (row < 0 || row >= features.length) return null;
    let best = null;
    let bestD = 36; // within 6px
    for (const d of dots) {
      if (d.row !== row) continue;
      const dist = (d.x - mx) ** 2 + (d.y - my) ** 2;
      if (dist < bestD) {
        bestD = dist;
        best = d;
      }
    }
    return best;
  };

  const f = hover ? features[hover.row] : null;
  const tx = hover ? transactions[hover.i] : null;
  return (
    <div className="beeswarm" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        style={{ width, height, cursor: hover ? "pointer" : "default" }}
        onMouseMove={(e) => setHover(findDot(e))}
        onMouseLeave={() => setHover(null)}
        onClick={() => hover && onSelect(tx.id)}
      />
      {hover && (
        <div className="tooltip beeswarm-tip" style={{ left: Math.min(hover.x + 12, width - 230), top: hover.y + 10 }}>
          <div className="tooltip-title">{f.feature}</div>
          <div className="tooltip-row"><span className="tooltip-label">SHAP</span><span className="tooltip-value">{hover.shap >= 0 ? "+" : ""}{fixed(hover.shap)}</span></div>
          <div className="tooltip-row"><span className="tooltip-label">Feature value</span><span className="tooltip-value">{hover.pct === null ? "missing" : `${Math.round(hover.pct * 100)}th pct`}</span></div>
          <div className="tooltip-row"><span className="tooltip-label">Transaction</span><span className="tooltip-value">{tx.id}</span></div>
          <div className="tooltip-hint">Click to explain this transaction</div>
        </div>
      )}
    </div>
  );
}

function niceStep(span) {
  const raw = span / 8;
  const mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw);
}
