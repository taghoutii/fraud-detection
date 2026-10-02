import { useEffect, useState } from "react";

// Read-only static JSON from public/data, fetched once per path and shared by every component.
const cache = new Map();

export function loadJSON(path) {
  if (!cache.has(path)) {
    cache.set(
      path,
      fetch(`${import.meta.env.BASE_URL}data/${path}`).then((r) => {
        if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
        return r.json();
      })
    );
  }
  return cache.get(path);
}

export function useData(path) {
  const [state, setState] = useState({ data: null, error: null });
  useEffect(() => {
    if (!path) return undefined;
    let live = true;
    setState({ data: null, error: null });
    loadJSON(path).then(
      (data) => live && setState({ data, error: null }),
      (error) => live && setState({ data: null, error })
    );
    return () => {
      live = false;
    };
  }, [path]);
  return state;
}

// ---------- formatting ----------
export const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}%`;
export const num = (x) => x.toLocaleString("en-US");
export const fixed = (x, digits = 3) => x.toFixed(digits);
export const signed = (x, digits = 3) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(digits)}`;
export const sigmoid = (z) => 1 / (1 + Math.exp(-z));

// fraud score: 3 decimals normally, 2 significant digits for tiny scores so they don't all read "0.000"
export function formatScore(p) {
  if (p >= 0.001 || p === 0) return p.toFixed(3);
  return p.toPrecision(2);
}

export function formatValue(v) {
  if (v === null || v === undefined) return "missing";
  if (Number.isInteger(v)) return num(v);
  return Math.abs(v) >= 1000 ? num(Math.round(v)) : Number(v.toPrecision(4)).toString();
}
