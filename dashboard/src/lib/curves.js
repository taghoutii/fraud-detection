// Client-side decision curves from the raw test predictions (probability + true label per row).
// Everything is exact over all 118k rows: sort once, then each threshold is a binary search.

export function preparePredictions({ proba, label }) {
  const n = proba.length;
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => proba[b] - proba[a]);
  const p = new Float64Array(n);
  const cumTP = new Int32Array(n + 1); // cumTP[k] = frauds among the k highest-scored rows
  for (let k = 0; k < n; k++) {
    p[k] = proba[order[k]];
    cumTP[k + 1] = cumTP[k] + label[order[k]];
  }
  const P = cumTP[n];
  return { n, p, cumTP, P, N: n - P };
}

// rows flagged at threshold t: proba >= t (same rule as Stage 6 and the API)
function countAtOrAbove(prep, t) {
  let lo = 0;
  let hi = prep.n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (prep.p[mid] >= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export function confusionAt(prep, t) {
  const flagged = countAtOrAbove(prep, t);
  const tp = prep.cumTP[flagged];
  const fp = flagged - tp;
  const fn = prep.P - tp;
  const tn = prep.N - fp;
  const precision = flagged ? tp / flagged : 0;
  const recall = tp / prep.P;
  return {
    tp, fp, fn, tn, flagged, precision, recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    fpr: fp / prep.N,
  };
}

// ROC + PR points at every distinct score, plus ROC-AUC (trapezoid) and average precision
// (sklearn's step definition), then thinned for plotting.
export function decisionCurves(prep, maxPoints = 400) {
  const { n, p, cumTP, P, N } = prep;
  const pts = [{ threshold: 1, fpr: 0, tpr: 0, precision: 1, recall: 0 }];
  let auc = 0;
  let ap = 0;
  let prevFpr = 0;
  let prevTpr = 0;
  for (let k = 1; k <= n; k++) {
    if (k < n && p[k] === p[k - 1]) continue; // only at the end of a run of tied scores
    const tp = cumTP[k];
    const fp = k - tp;
    const fpr = fp / N;
    const tpr = tp / P;
    const precision = tp / k;
    auc += ((fpr - prevFpr) * (tpr + prevTpr)) / 2;
    ap += (tpr - prevTpr) * precision;
    pts.push({ threshold: p[k - 1], fpr, tpr, precision, recall: tpr });
    prevFpr = fpr;
    prevTpr = tpr;
  }
  // thin evenly along the curve, always keeping both ends
  const step = Math.max(1, Math.ceil(pts.length / maxPoints));
  const thin = pts.filter((_, i) => i % step === 0 || i === pts.length - 1);
  return { points: thin, rocAuc: auc, averagePrecision: ap };
}
