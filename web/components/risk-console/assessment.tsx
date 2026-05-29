import { AlertTriangle, BarChart3, Brain, CheckCircle2 } from "lucide-react";
import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { featureLabel, formatFeatureValue } from "@/lib/labels";
import type { PredictionResult } from "@/types/healthcare";

import { DataTable, NarrativePanel, PanelHeading } from "./shared-ui";
import type { ChartFeature } from "./types";

export function Assessment({ result }: { result: PredictionResult }) {
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
              contentStyle={{
                background: "#101318",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 8,
              }}
            />
            <ReferenceLine x={0} stroke="rgba(244,246,248,0.45)" strokeDasharray="3 3" />
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
