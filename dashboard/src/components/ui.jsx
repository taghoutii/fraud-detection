// Small presentational building blocks shared by every section.

export function Card({ title, sub, action, children, className = "" }) {
  return (
    <div className={`card ${className}`}>
      {(title || action) && (
        <div className="card-head">
          <div>
            {title && <h2 className="card-title">{title}</h2>}
            {sub && <p className="card-sub">{sub}</p>}
          </div>
          {action}
        </div>
      )}
      {children}
    </div>
  );
}

export function Loading({ height = 240, error }) {
  return (
    <div className={`loading ${error ? "loading-error" : ""}`} style={{ height }}>
      {error ? `Couldn't load data: ${error.message}` : null}
    </div>
  );
}

export function Stat({ label, value, hint, tone }) {
  return (
    <div className={`stat ${tone ? `stat-${tone}` : ""}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

// A confident, prominent insight card (not a caveat box)
export function Insight({ eyebrow, title, children, metrics, icon = "spark" }) {
  return (
    <div className="insight">
      <div className="insight-icon" aria-hidden="true">{ICONS[icon]}</div>
      <div className="insight-body">
        {eyebrow && <div className="insight-eyebrow">{eyebrow}</div>}
        <h3 className="insight-title">{title}</h3>
        <p className="insight-text">{children}</p>
        {metrics && (
          <div className="insight-metrics">
            {metrics.map((m) => (
              <div key={m.label} className="insight-metric">
                <span className="insight-metric-value">{m.value}</span>
                <span className="insight-metric-label">{m.label}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="tablist" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          aria-selected={value === o.value}
          className="segmented-option"
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// Recharts tooltip body: rows of [label, value, swatch color]
export function TooltipBox({ title, rows }) {
  return (
    <div className="tooltip">
      {title && <div className="tooltip-title">{title}</div>}
      {rows.map(([label, value, color]) => (
        <div key={label} className="tooltip-row">
          {color && <span className="tooltip-swatch" style={{ background: color }} />}
          <span className="tooltip-label">{label}</span>
          <span className="tooltip-value">{value}</span>
        </div>
      ))}
    </div>
  );
}

export function LegendItem({ color, label, dashed }) {
  return (
    <span className="legend-item">
      <span className={`legend-swatch ${dashed ? "legend-swatch-dashed" : ""}`} style={{ "--swatch": color }} />
      {label}
    </span>
  );
}

const ICONS = {
  spark: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6.3 6.3l2.8 2.8M14.9 14.9l2.8 2.8M6.3 17.7l2.8-2.8M14.9 9.1l2.8-2.8" />
    </svg>
  ),
  trend: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M3 17l6-6 4 4 8-8" />
      <path d="M14 7h7v7" />
    </svg>
  ),
  gauge: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M4 18a8 8 0 1 1 16 0" />
      <path d="M12 18l4-6" />
    </svg>
  ),
  trophy: (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <path d="M8 21h8M12 17v4M7 4h10v5a5 5 0 0 1-10 0V4z" />
      <path d="M17 6h3a3 3 0 0 1-3 4M7 6H4a3 3 0 0 0 3 4" />
    </svg>
  ),
};
