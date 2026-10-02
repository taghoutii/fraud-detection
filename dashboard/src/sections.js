// Nav order is the story: which model -> how probabilities become decisions -> why this transaction.
export const SECTIONS = [
  { id: "overview", step: "01", label: "Overview", question: "System at a glance" },
  { id: "model-selection", step: "02", label: "Model Selection", question: "Which model did I choose?" },
  { id: "decisioning", step: "03", label: "Decisioning", question: "How do probabilities become decisions?" },
  { id: "explainability", step: "04", label: "Explainability", question: "Why did it flag this transaction?" },
];
