export const FEATURE_LABELS: Record<string, string> = {
  HighBP: "High blood pressure",
  HighChol: "High cholesterol",
  CholCheck: "Cholesterol checked",
  BMI: "BMI",
  Smoker: "Smoker",
  Stroke: "Stroke history",
  HeartDiseaseorAttack: "Heart disease history",
  PhysActivity: "Physical activity",
  Fruits: "Fruit intake",
  Veggies: "Vegetable intake",
  HvyAlcoholConsump: "Heavy alcohol use",
  AnyHealthcare: "Healthcare coverage",
  NoDocbcCost: "Cost barrier",
  GenHlth: "General health",
  MentHlth: "Mental health days",
  PhysHlth: "Physical health days",
  DiffWalk: "Difficulty walking",
  Sex: "Sex",
  Age: "Age band",
  Education: "Education level",
  Income: "Income level",
  age: "Age",
  bmi: "BMI",
  bp: "Blood pressure",
  glucose: "Glucose",
};

export const BINARY_FEATURES = new Set([
  "HighBP",
  "HighChol",
  "CholCheck",
  "Smoker",
  "Stroke",
  "HeartDiseaseorAttack",
  "PhysActivity",
  "Fruits",
  "Veggies",
  "HvyAlcoholConsump",
  "AnyHealthcare",
  "NoDocbcCost",
  "DiffWalk",
]);

export const ESCALATION_LABELS: Record<string, string> = {
  routine_followup: "Routine follow-up",
  prompt_clinician_followup: "Prompt clinician follow-up",
  urgent_clinician_review: "Urgent clinician review",
};

export const RISK_COLORS: Record<string, string> = {
  Low: "#2f9b6a",
  Medium: "#d49a35",
  High: "#d05245",
};

export const ESCALATION_COLORS: Record<string, string> = {
  routine_followup: "#2f9b6a",
  prompt_clinician_followup: "#d49a35",
  urgent_clinician_review: "#d05245",
};

const GENERAL_HEALTH_LABELS: Record<number, string> = {
  1: "Excellent",
  2: "Very good",
  3: "Good",
  4: "Fair",
  5: "Poor",
};

export function featureLabel(feature: string) {
  return FEATURE_LABELS[feature] ?? feature;
}

export function formatFeatureValue(feature: string, value: number) {
  if (BINARY_FEATURES.has(feature)) {
    return Math.round(value) === 1 ? "Yes" : "No";
  }
  if (feature === "Sex") {
    return Math.round(value) === 1 ? "Male" : "Female";
  }
  if (feature === "GenHlth") {
    return GENERAL_HEALTH_LABELS[Math.round(value)] ?? String(value);
  }
  if (feature === "BMI") {
    return value.toFixed(1);
  }
  if (["Age", "Education", "Income", "MentHlth", "PhysHlth"].includes(feature)) {
    return String(Math.round(value));
  }
  return value.toFixed(2);
}

export function formatModelName(modelName?: string) {
  if (!modelName) return "Unknown";
  return modelName
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
