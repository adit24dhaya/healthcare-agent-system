"use client";

import { QueryClient, QueryClientProvider, useMutation, useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  CheckCircle2,
  ClipboardList,
  Database,
  Gauge,
  HeartPulse,
  Loader2,
  ShieldCheck,
  Sparkles,
  Stethoscope,
} from "lucide-react";
import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ReactNode } from "react";

import { createPrediction, getHistory } from "@/lib/api";
import {
  ESCALATION_COLORS,
  ESCALATION_LABELS,
  RISK_COLORS,
  featureLabel,
  formatFeatureValue,
  formatModelName,
} from "@/lib/labels";
import type { FeatureImpact, PatientInput, PredictionResult, RiskLevel } from "@/types/healthcare";

const queryClient = new QueryClient();

const DEFAULT_INPUT: PatientInput = {
  age: 45,
  height_cm: 170,
  weight_kg: 82.4,
  bp: 130,
  glucose: 180,
  high_chol: false,
  chol_check: true,
  smoker: false,
  stroke: false,
  heart_disease_or_attack: false,
  phys_activity: true,
  fruits: true,
  veggies: true,
  heavy_alcohol_consump: false,
  any_healthcare: true,
  no_doc_bc_cost: false,
  general_health: 3,
  mental_health_days: 2,
  physical_health_days: 2,
  diff_walk: false,
  sex: "female",
  education: 5,
  income: 5,
};

type TabKey = "assessment" | "evidence" | "history";

export function RiskConsole() {
  return (
    <QueryClientProvider client={queryClient}>
      <RiskConsoleInner />
    </QueryClientProvider>
  );
}

function RiskConsoleInner() {
  const [form, setForm] = useState<PatientInput>(DEFAULT_INPUT);
  const [activeTab, setActiveTab] = useState<TabKey>("assessment");
  const [lastResult, setLastResult] = useState<PredictionResult | null>(null);

  const mutation = useMutation({
    mutationFn: createPrediction,
    onSuccess: (result) => {
      setLastResult(result);
      setActiveTab("assessment");
    },
  });

  const result = mutation.data ?? lastResult;
  const historyQuery = useQuery({
    queryKey: ["history", result?.request_id],
    queryFn: () => getHistory(25),
    enabled: Boolean(result),
  });
  const calculatedBmi = calculateBmi(form.height_cm, form.weight_kg);

  function updateField<K extends keyof PatientInput>(key: K, value: PatientInput[K]) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function submit() {
    mutation.mutate(form);
  }

  return (
    <main className="min-h-screen bg-canvas text-ink">
      <div className="grid min-h-screen grid-cols-1 xl:grid-cols-[420px_1fr]">
        <aside className="border-b border-borderSoft bg-panelMuted/80 px-5 py-6 xl:h-screen xl:overflow-y-auto xl:border-b-0 xl:border-r">
          <div className="mb-6 flex items-start justify-between gap-4">
            <div>
              <div className="flex items-center gap-2 text-sm font-semibold text-info">
                <Stethoscope className="h-4 w-4" />
                Patient Intake
              </div>
              <h1 className="mt-2 text-2xl font-semibold tracking-tight">Healthcare AI Risk Console</h1>
            </div>
            <Pill tone="neutral">Local</Pill>
          </div>

          <InputSection title="Vitals" icon={<HeartPulse className="h-4 w-4" />}>
            <NumberField
              label="Age"
              min={0}
              max={120}
              value={form.age}
              onChange={(value) => updateField("age", value)}
            />
            <SelectField
              label="Sex"
              value={form.sex}
              options={[
                { value: "female", label: "Female" },
                { value: "male", label: "Male" },
              ]}
              onChange={(value) => updateField("sex", value as PatientInput["sex"])}
            />
            <div className="grid grid-cols-2 gap-3">
              <NumberField
                label="Height"
                suffix="cm"
                min={50}
                max={250}
                step={0.5}
                value={form.height_cm}
                onChange={(value) => updateField("height_cm", value)}
              />
              <NumberField
                label="Weight"
                suffix="kg"
                min={2}
                max={300}
                step={0.1}
                value={form.weight_kg}
                onChange={(value) => updateField("weight_kg", value)}
              />
            </div>
            <div className="grid grid-cols-2 gap-3">
              <NumberField
                label="Systolic BP"
                min={0}
                max={260}
                value={form.bp}
                onChange={(value) => updateField("bp", value)}
              />
              <NumberField
                label="Glucose"
                suffix="mg/dL"
                min={0}
                max={500}
                value={form.glucose ?? 0}
                onChange={(value) => updateField("glucose", value)}
              />
            </div>
            <MiniStat label="Calculated BMI" value={calculatedBmi.toFixed(1)} />
          </InputSection>

          <InputSection title="Health Profile" icon={<ClipboardList className="h-4 w-4" />}>
            <ToggleGrid>
              <Toggle
                label="High cholesterol"
                checked={form.high_chol}
                onChange={(value) => updateField("high_chol", value)}
              />
              <Toggle
                label="Cholesterol checked"
                checked={form.chol_check}
                onChange={(value) => updateField("chol_check", value)}
              />
              <Toggle
                label="Smoker"
                checked={form.smoker}
                onChange={(value) => updateField("smoker", value)}
              />
              <Toggle
                label="Stroke history"
                checked={form.stroke}
                onChange={(value) => updateField("stroke", value)}
              />
              <Toggle
                label="Heart disease"
                checked={form.heart_disease_or_attack}
                onChange={(value) => updateField("heart_disease_or_attack", value)}
              />
              <Toggle
                label="Difficulty walking"
                checked={form.diff_walk}
                onChange={(value) => updateField("diff_walk", value)}
              />
            </ToggleGrid>
            <SliderField
              label="General health"
              min={1}
              max={5}
              value={form.general_health}
              onChange={(value) => updateField("general_health", value)}
            />
            <SliderField
              label="Mental health days"
              min={0}
              max={30}
              value={form.mental_health_days}
              onChange={(value) => updateField("mental_health_days", value)}
            />
            <SliderField
              label="Physical health days"
              min={0}
              max={30}
              value={form.physical_health_days}
              onChange={(value) => updateField("physical_health_days", value)}
            />
          </InputSection>

          <InputSection title="Lifestyle and Access" icon={<Activity className="h-4 w-4" />}>
            <ToggleGrid>
              <Toggle
                label="Physical activity"
                checked={form.phys_activity}
                onChange={(value) => updateField("phys_activity", value)}
              />
              <Toggle
                label="Fruit intake"
                checked={form.fruits}
                onChange={(value) => updateField("fruits", value)}
              />
              <Toggle
                label="Vegetable intake"
                checked={form.veggies}
                onChange={(value) => updateField("veggies", value)}
              />
              <Toggle
                label="Heavy alcohol"
                checked={form.heavy_alcohol_consump}
                onChange={(value) => updateField("heavy_alcohol_consump", value)}
              />
              <Toggle
                label="Healthcare coverage"
                checked={form.any_healthcare}
                onChange={(value) => updateField("any_healthcare", value)}
              />
              <Toggle
                label="Cost barrier"
                checked={form.no_doc_bc_cost}
                onChange={(value) => updateField("no_doc_bc_cost", value)}
              />
            </ToggleGrid>
            <SliderField
              label="Education level"
              min={1}
              max={6}
              value={form.education}
              onChange={(value) => updateField("education", value)}
            />
            <SliderField
              label="Income level"
              min={1}
              max={8}
              value={form.income}
              onChange={(value) => updateField("income", value)}
            />
          </InputSection>

          <button
            type="button"
            onClick={submit}
            disabled={mutation.isPending}
            className="mt-4 flex h-12 w-full items-center justify-center gap-2 rounded-md bg-info px-4 text-sm font-bold text-canvas transition hover:bg-[#7bbce9] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {mutation.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            Analyze Patient
          </button>

          {mutation.error ? (
            <div className="mt-4 rounded-md border border-danger/40 bg-danger/10 p-3 text-sm text-[#ffb4ad]">
              {mutation.error.message}
            </div>
          ) : null}
        </aside>

        <section className="px-5 py-6 md:px-8 lg:px-10">
          <div className="mx-auto flex max-w-7xl flex-col gap-6">
            <Hero result={result} />
            {result ? (
              <>
                <ModelStrip result={result} />
                <Tabs activeTab={activeTab} onChange={setActiveTab} />
                {activeTab === "assessment" ? <Assessment result={result} /> : null}
                {activeTab === "evidence" ? <Evidence result={result} /> : null}
                {activeTab === "history" ? (
                  <History
                    result={result}
                    records={historyQuery.data?.records ?? []}
                    isLoading={historyQuery.isLoading}
                  />
                ) : null}
              </>
            ) : (
              <EmptyState />
            )}
          </div>
        </section>
      </div>
    </main>
  );
}

function Hero({ result }: { result: PredictionResult | null | undefined }) {
  const risk = result?.risk ?? "Low";
  const riskColor = RISK_COLORS[risk];
  const confidenceLabel = result?.safety?.confidence_label ?? "Waiting";
  const confidenceScore = result?.safety?.confidence_score;
  const confidence =
    confidenceScore !== undefined
      ? `${confidenceLabel} (${(confidenceScore * 100).toFixed(0)}%)`
      : confidenceLabel;
  const escalation = result?.safety?.escalation ?? "routine_followup";
  const escalationLabel = ESCALATION_LABELS[escalation] ?? escalation;

  return (
    <header className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
      <div>
        <div className="flex items-center gap-2 text-sm font-semibold text-info">
          <Brain className="h-4 w-4" />
          AI/ML Decision Support
        </div>
        <h2 className="mt-3 max-w-3xl text-4xl font-semibold tracking-tight md:text-5xl">
          Healthcare AI Risk Console
        </h2>
        <p className="mt-3 max-w-3xl text-sm leading-6 text-muted">
          Educational prototype only; not medical advice. Clinical decisions require qualified
          clinician review and measured labs/vitals.
        </p>
      </div>

      <div className="console-panel grid gap-3 rounded-lg p-4 sm:grid-cols-2">
        <MetricTile
          label="Risk"
          value={result ? result.risk : "Not run"}
          icon={<Gauge className="h-4 w-4" />}
          accent={riskColor}
        />
        <MetricTile
          label="Probability"
          value={result ? `${(result.probability * 100).toFixed(1)}%` : "—"}
          icon={<BarChart3 className="h-4 w-4" />}
        />
        <MetricTile label="BMI" value={result ? result.patient.bmi.toFixed(1) : "—"} />
        <MetricTile label="Confidence" value={confidence} icon={<ShieldCheck className="h-4 w-4" />} />
        <div className="sm:col-span-2">
          <div className="rounded-md border border-borderSoft bg-canvas/50 p-3">
            <div className="field-label">Escalation</div>
            <div className="mt-2 flex items-center gap-2">
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: ESCALATION_COLORS[escalation] }}
              />
              <span className="text-sm font-semibold text-ink">{escalationLabel}</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}

function ModelStrip({ result }: { result: PredictionResult }) {
  const model = result.model;
  const selectedModel = model?.selected_model;
  const metrics = selectedModel ? model?.metrics?.[selectedModel] : undefined;

  return (
    <section className="console-panel grid gap-3 rounded-lg p-4 md:grid-cols-5">
      <ModelItem label="Model" value={formatModelName(selectedModel)} />
      <ModelItem label="Rows" value={model?.rows_total ? model.rows_total.toLocaleString() : "—"} />
      <ModelItem label="ROC AUC" value={metrics?.roc_auc ? metrics.roc_auc.toFixed(3) : "—"} />
      <ModelItem label="Calibration" value={model?.calibration ?? "—"} />
      <ModelItem label="Source" value={model?.dataset_slug ?? "Kaggle artifact"} />
    </section>
  );
}

function Assessment({ result }: { result: PredictionResult }) {
  const topFeatures = useMemo(() => {
    return result.feature_explanation.features
      .slice()
      .sort((a, b) => b.magnitude - a.magnitude)
      .slice(0, 8)
      .map((feature) => ({
        ...feature,
        label: featureLabel(feature.feature),
        valueLabel: formatFeatureValue(feature.feature, feature.value),
        directionLabel: feature.impact > 0 ? "Raises risk" : "Lowers risk",
      }));
  }, [result.feature_explanation.features]);

  return (
    <section className="grid gap-5 xl:grid-cols-[0.95fr_1.05fr]">
      <div className="space-y-5">
        <NarrativePanel title="Explanation" icon={<Brain className="h-4 w-4" />} text={result.explanation} />
        <NarrativePanel
          title="Recommendation"
          icon={<CheckCircle2 className="h-4 w-4" />}
          text={result.recommendation}
        />
        {result.safety.alerts.length > 0 ? (
          <div className="rounded-lg border border-danger/40 bg-danger/10 p-4 text-sm text-[#ffc1bc]">
            <div className="mb-2 flex items-center gap-2 font-semibold text-ink">
              <AlertTriangle className="h-4 w-4 text-danger" />
              Safety Alerts
            </div>
            <ul className="space-y-1">
              {result.safety.alerts.map((alert) => (
                <li key={alert}>{alert}</li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
      <FeatureImpactPanel features={topFeatures} method={result.feature_explanation.method} />
    </section>
  );
}

function Evidence({ result }: { result: PredictionResult }) {
  const contexts = result.retrieved_context ?? [];

  return (
    <section className="grid gap-5 xl:grid-cols-[1fr_0.9fr]">
      <div className="console-panel rounded-lg p-5">
        <PanelHeading icon={<Database className="h-4 w-4" />} title="Retrieved Medical Context" />
        <div className="mt-4 space-y-3">
          {contexts.length > 0 ? (
            contexts.map((context) => (
              <div key={context.title} className="rounded-md border border-borderSoft bg-canvas/40 p-4">
                <div className="font-semibold text-ink">{context.title}</div>
                <p className="mt-2 text-sm leading-6 text-muted">{context.text}</p>
              </div>
            ))
          ) : (
            <p className="text-sm text-muted">No medical context retrieved for this case.</p>
          )}
        </div>
      </div>
      <div className="console-panel rounded-lg p-5">
        <PanelHeading icon={<Activity className="h-4 w-4" />} title="Similar Prior Cases" />
        <DataTable
          rows={result.similar_cases.map((item) => ({
            Risk: item.metadata.risk ?? "—",
            Probability:
              typeof item.metadata.probability === "number"
                ? `${(item.metadata.probability * 100).toFixed(1)}%`
                : "—",
            Age: item.metadata.age ?? "—",
            BMI: item.metadata.bmi ?? "—",
            BP: item.metadata.bp ?? "—",
            Distance: item.distance.toFixed(3),
          }))}
        />
      </div>
    </section>
  );
}

function History({
  result,
  records,
  isLoading,
}: {
  result: PredictionResult;
  records: Array<{
    id: string;
    summary: string;
    metadata: {
      timestamp?: string;
      risk?: RiskLevel;
      probability?: number;
      age?: number;
      bmi?: number;
      bp?: number;
      glucose?: number;
    };
  }>;
  isLoading: boolean;
}) {
  const fallbackRows = result.similar_cases.map((item, index) => ({
    id: index + 1,
    risk: item.metadata.risk ?? "Unknown",
    probability: item.metadata.probability ?? 0,
    timestamp: item.metadata.timestamp ?? "Stored memory",
  }));
  const rows =
    records.length > 0
      ? records.map((item, index) => ({
          id: index + 1,
          risk: item.metadata.risk ?? "Unknown",
          probability: item.metadata.probability ?? 0,
          timestamp: item.metadata.timestamp ?? item.summary,
        }))
      : fallbackRows;

  return (
    <section className="grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
      <div className="console-panel rounded-lg p-5">
        <PanelHeading icon={<BarChart3 className="h-4 w-4" />} title="Risk Distribution" />
        <div className="mt-4 h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={riskDistribution(rows)} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
              <XAxis dataKey="risk" stroke="#9aa4b2" tickLine={false} axisLine={false} />
              <YAxis stroke="#9aa4b2" tickLine={false} axisLine={false} />
              <Tooltip
                cursor={{ fill: "rgba(255,255,255,0.04)" }}
                contentStyle={{
                  background: "#101318",
                  border: "1px solid rgba(255,255,255,0.1)",
                  borderRadius: 8,
                }}
              />
              <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                {riskDistribution(rows).map((entry) => (
                  <Cell key={entry.risk} fill={RISK_COLORS[entry.risk] ?? "#5aa3d9"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="console-panel rounded-lg p-5">
        <PanelHeading icon={<Database className="h-4 w-4" />} title="Memory Records" />
        {isLoading ? <p className="mt-4 text-sm text-muted">Loading memory records...</p> : null}
        <DataTable
          rows={rows.map((row) => ({
            Case: row.id,
            Risk: row.risk,
            Probability: `${(row.probability * 100).toFixed(1)}%`,
            Timestamp: row.timestamp,
          }))}
        />
      </div>
    </section>
  );
}

type ChartFeature = FeatureImpact & {
  label: string;
  valueLabel: string;
  directionLabel: string;
};

function FeatureImpactPanel({ features, method }: { features: ChartFeature[]; method: string }) {
  return (
    <div className="console-panel rounded-lg p-5">
      <PanelHeading icon={<BarChart3 className="h-4 w-4" />} title="Feature Impact" />
      <div className="mt-4 h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={features} layout="vertical" margin={{ top: 8, right: 18, bottom: 8, left: 28 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.08)" horizontal={false} />
            <XAxis type="number" stroke="#9aa4b2" tickLine={false} axisLine={false} />
            <YAxis
              type="category"
              dataKey="label"
              width={135}
              stroke="#9aa4b2"
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 12 }}
            />
            <Tooltip
              cursor={{ fill: "rgba(255,255,255,0.04)" }}
              formatter={(value) => [Number(value).toFixed(4), "Impact"]}
              labelFormatter={(label) => label}
              contentStyle={{
                background: "#101318",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 8,
              }}
            />
            <Bar dataKey="impact" radius={[0, 4, 4, 0]}>
              {features.map((feature) => (
                <Cell key={feature.feature} fill={feature.impact >= 0 ? "#d05245" : "#2f9b6a"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 flex items-center justify-between gap-3 text-xs text-muted">
        <span>Method: {method}</span>
        <span>Top {features.length} by absolute impact</span>
      </div>
      <DataTable
        rows={features.map((feature) => ({
          Feature: feature.label,
          Value: feature.valueLabel,
          Direction: feature.directionLabel,
          Impact: feature.impact.toFixed(4),
        }))}
        compact
      />
    </div>
  );
}

function Tabs({ activeTab, onChange }: { activeTab: TabKey; onChange: (tab: TabKey) => void }) {
  const tabs: Array<{ key: TabKey; label: string }> = [
    { key: "assessment", label: "Assessment" },
    { key: "evidence", label: "Evidence" },
    { key: "history", label: "History" },
  ];

  return (
    <div className="flex flex-wrap gap-2 border-b border-borderSoft pb-3">
      {tabs.map((tab) => (
        <button
          key={tab.key}
          type="button"
          onClick={() => onChange(tab.key)}
          className={clsx(
            "rounded-md px-4 py-2 text-sm font-semibold transition",
            activeTab === tab.key
              ? "bg-info text-canvas"
              : "border border-borderSoft bg-panel text-muted hover:text-ink",
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

function EmptyState() {
  return (
    <div className="console-panel rounded-lg p-8">
      <div className="flex max-w-2xl items-start gap-4">
        <div className="rounded-md border border-borderSoft bg-info/10 p-3 text-info">
          <Sparkles className="h-5 w-5" />
        </div>
        <div>
          <h3 className="text-xl font-semibold">Ready for assessment</h3>
          <p className="mt-2 text-sm leading-6 text-muted">
            Fill in the patient profile and run the trained Kaggle artifact through the FastAPI
            inference service.
          </p>
        </div>
      </div>
    </div>
  );
}

function InputSection({
  title,
  icon,
  children,
}: {
  title: string;
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="mb-4 rounded-lg border border-borderSoft bg-panel p-4">
      <div className="mb-4 flex items-center gap-2 text-sm font-semibold text-ink">
        <span className="text-info">{icon}</span>
        {title}
      </div>
      <div className="space-y-3">{children}</div>
    </section>
  );
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  suffix,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step?: number;
  suffix?: string;
}) {
  return (
    <label className="block">
      <div className="mb-1 flex items-center justify-between gap-2">
        <span className="field-label">{label}</span>
        {suffix ? <span className="text-xs text-muted">{suffix}</span> : null}
      </div>
      <input
        className="field-input"
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="field-label mb-1 block">{label}</span>
      <select className="field-input" value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function SliderField({
  label,
  value,
  onChange,
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
}) {
  return (
    <label className="block rounded-md border border-borderSoft bg-canvas/50 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="field-label">{label}</span>
        <span className="text-sm font-semibold text-info">{value}</span>
      </div>
      <input
        className="w-full"
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <div className="mt-1 flex justify-between text-[11px] text-muted">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </label>
  );
}

function Toggle({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="toggle-row">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="h-4 w-4 accent-info"
      />
      <span>{label}</span>
    </label>
  );
}

function ToggleGrid({ children }: { children: ReactNode }) {
  return <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">{children}</div>;
}

function MetricTile({
  label,
  value,
  icon,
  accent = "#5aa3d9",
}: {
  label: string;
  value: string;
  icon?: ReactNode;
  accent?: string;
}) {
  return (
    <div className="rounded-md border border-borderSoft bg-canvas/50 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="field-label">{label}</div>
        <span style={{ color: accent }}>{icon}</span>
      </div>
      <div className="mt-2 text-2xl font-semibold tracking-tight">{value}</div>
    </div>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-borderSoft bg-canvas/50 p-3">
      <div className="field-label">{label}</div>
      <div className="mt-1 text-lg font-semibold text-ink">{value}</div>
    </div>
  );
}

function ModelItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="field-label">{label}</div>
      <div className="mt-1 truncate text-sm font-semibold text-ink" title={value}>
        {value}
      </div>
    </div>
  );
}

function NarrativePanel({
  title,
  icon,
  text,
}: {
  title: string;
  icon: ReactNode;
  text: string;
}) {
  return (
    <div className="console-panel rounded-lg p-5">
      <PanelHeading icon={icon} title={title} />
      <p className="mt-4 text-sm leading-7 text-ink/90">{text}</p>
    </div>
  );
}

function PanelHeading({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="flex items-center gap-2 text-lg font-semibold">
      <span className="text-info">{icon}</span>
      {title}
    </div>
  );
}

function Pill({ children, tone }: { children: ReactNode; tone: "neutral" }) {
  return (
    <span
      className={clsx(
        "rounded-full border px-3 py-1 text-xs font-semibold",
        tone === "neutral" && "border-borderSoft bg-canvas/70 text-muted",
      )}
    >
      {children}
    </span>
  );
}

function DataTable({
  rows,
  compact = false,
}: {
  rows: Array<Record<string, string | number>>;
  compact?: boolean;
}) {
  const columns = rows[0] ? Object.keys(rows[0]) : [];

  if (!rows.length) {
    return <p className="mt-4 text-sm text-muted">No records available.</p>;
  }

  return (
    <div className={clsx("overflow-hidden rounded-md border border-borderSoft", compact ? "mt-4" : "mt-4")}>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] border-collapse text-left text-sm">
          <thead className="bg-panelMuted text-xs uppercase tracking-[0.08em] text-muted">
            <tr>
              {columns.map((column) => (
                <th key={column} className="border-b border-borderSoft px-3 py-2 font-semibold">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={index} className="border-b border-borderSoft last:border-b-0">
                {columns.map((column) => (
                  <td key={column} className="px-3 py-2 text-ink/90">
                    {row[column]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function calculateBmi(heightCm: number, weightKg: number) {
  const heightM = heightCm / 100;
  if (heightM <= 0) return 0;
  return weightKg / (heightM * heightM);
}

function riskDistribution(rows: Array<{ risk: string }>) {
  const counts: Record<RiskLevel, number> = { Low: 0, Medium: 0, High: 0 };
  rows.forEach((row) => {
    if (row.risk === "Low" || row.risk === "Medium" || row.risk === "High") {
      counts[row.risk] += 1;
    }
  });
  return Object.entries(counts).map(([risk, count]) => ({ risk: risk as RiskLevel, count }));
}
