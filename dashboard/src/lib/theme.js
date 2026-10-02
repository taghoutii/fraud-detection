// Chart colors, read from the CSS custom properties in tokens.css so the palette has one
// source of truth. SVG presentation attributes can't use var(), so charts get resolved hex
// values (read lazily, at render time, after the stylesheet has loaded).
let cached = null;

export function palette() {
  if (cached) return cached;
  const css = getComputedStyle(document.documentElement);
  const read = (name) => css.getPropertyValue(name).trim();
  cached = {
    deepRed: read("--palette-deep-red"), // confirmed fraud / high risk
    red: read("--palette-red"), // flagged / alerts / pushes toward fraud
    cream: read("--palette-cream"),
    navy: read("--palette-navy"),
    steel: read("--palette-steel"), // safe / neutral data
  };
  return cached;
}

// axis + grid styling shared by every Recharts chart
export function axisProps() {
  const p = palette();
  return {
    stroke: p.navy,
    strokeOpacity: 0.25,
    tick: { fill: p.navy, fillOpacity: 0.65, fontSize: 12 },
    tickLine: false,
  };
}

export function gridProps() {
  return { stroke: palette().navy, strokeOpacity: 0.08, vertical: false };
}

// linear blend between two hex colors, t in [0, 1]
export function mix(hexA, hexB, t) {
  const a = parseInt(hexA.slice(1), 16);
  const b = parseInt(hexB.slice(1), 16);
  const ch = (shift) => Math.round(((a >> shift) & 255) * (1 - t) + ((b >> shift) & 255) * t);
  return `rgb(${ch(16)}, ${ch(8)}, ${ch(0)})`;
}
