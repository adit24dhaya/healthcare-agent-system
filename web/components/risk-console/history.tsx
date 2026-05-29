import { BarChart3, Database } from "lucide-react";
import { useMemo } from "react";
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

import { RISK_COLORS } from "@/lib/labels";
import type { MemoryRecord, PredictionResult } from "@/types/healthcare";

import { DataTable, PanelHeading } from "./shared-ui";
import { formatMemoryTimestamp, riskDistribution } from "./utils";

export function History({
  result,
  records,
  isLoading,
}: {
  result: PredictionResult;
  records: MemoryRecord[];
  isLoading: boolean;
}) {
  const fallbackRows = useMemo(
    () =>
      result.similar_cases.map((item, index) => ({
        id: index + 1,
        risk: item.metadata.risk ?? "Unknown",
        probability: item.metadata.probability ?? 0,
        timestamp: formatMemoryTimestamp(item.metadata.timestamp, "Stored memory"),
      })),
    [result.similar_cases],
  );
  const rows = useMemo(
    () =>
      records.length > 0
        ? records.map((item, index) => ({
            id: index + 1,
            risk: item.metadata.risk ?? "Unknown",
            probability: item.metadata.probability ?? 0,
            timestamp: formatMemoryTimestamp(item.metadata.timestamp, item.summary),
          }))
        : fallbackRows,
    [fallbackRows, records],
  );
  const distribution = useMemo(() => riskDistribution(rows), [rows]);

  return (
    <section className="grid gap-5 xl:grid-cols-[0.9fr_1.1fr]">
      <div className="console-panel rounded-lg p-5">
        <PanelHeading icon={<BarChart3 className="h-4 w-4" />} title="Risk Distribution" />
        <div className="mt-4 h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={distribution} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
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
                {distribution.map((entry) => (
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
