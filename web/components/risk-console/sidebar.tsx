import {
  Activity,
  ClipboardList,
  HeartPulse,
  Loader2,
  Sparkles,
  Stethoscope,
} from "lucide-react";

import type { PatientInput } from "@/types/healthcare";

import type { PatientFieldChange } from "./types";
import {
  GlucoseField,
  InputSection,
  NumberField,
  SelectField,
  SliderField,
  Toggle,
  ToggleGrid,
} from "./form-controls";
import { MiniStat, Pill } from "./shared-ui";

export function PatientSidebar({
  form,
  calculatedBmi,
  isPending,
  error,
  onSubmit,
  onFieldChange,
}: {
  form: PatientInput;
  calculatedBmi: number;
  isPending: boolean;
  error: Error | null;
  onSubmit: () => void;
  onFieldChange: PatientFieldChange;
}) {
  return (
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
          onChange={(value) => onFieldChange("age", value)}
        />
        <SelectField
          label="Sex"
          value={form.sex}
          options={[
            { value: "female", label: "Female" },
            { value: "male", label: "Male" },
          ]}
          onChange={(value) => onFieldChange("sex", value as PatientInput["sex"])}
        />
        <div className="grid grid-cols-2 gap-3">
          <NumberField
            label="Height"
            suffix="cm"
            min={50}
            max={250}
            step={0.5}
            value={form.height_cm}
            onChange={(value) => onFieldChange("height_cm", value)}
          />
          <NumberField
            label="Weight"
            suffix="kg"
            min={2}
            max={300}
            step={0.1}
            value={form.weight_kg}
            onChange={(value) => onFieldChange("weight_kg", value)}
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <NumberField
            label="Systolic BP"
            min={0}
            max={260}
            value={form.bp}
            onChange={(value) => onFieldChange("bp", value)}
          />
          <GlucoseField value={form.glucose} onChange={(value) => onFieldChange("glucose", value)} />
        </div>
        <MiniStat label="Calculated BMI" value={calculatedBmi.toFixed(1)} />
      </InputSection>

      <InputSection title="Health Profile" icon={<ClipboardList className="h-4 w-4" />} defaultOpen={false}>
        <ToggleGrid>
          <Toggle
            label="High cholesterol"
            checked={form.high_chol}
            onChange={(value) => onFieldChange("high_chol", value)}
          />
          <Toggle
            label="Cholesterol checked"
            checked={form.chol_check}
            onChange={(value) => onFieldChange("chol_check", value)}
          />
          <Toggle label="Smoker" checked={form.smoker} onChange={(value) => onFieldChange("smoker", value)} />
          <Toggle
            label="Stroke history"
            checked={form.stroke}
            onChange={(value) => onFieldChange("stroke", value)}
          />
          <Toggle
            label="Heart disease"
            checked={form.heart_disease_or_attack}
            onChange={(value) => onFieldChange("heart_disease_or_attack", value)}
          />
          <Toggle
            label="Difficulty walking"
            checked={form.diff_walk}
            onChange={(value) => onFieldChange("diff_walk", value)}
          />
        </ToggleGrid>
        <SliderField
          label="General health"
          min={1}
          max={5}
          value={form.general_health}
          onChange={(value) => onFieldChange("general_health", value)}
        />
        <SliderField
          label="Mental health days"
          min={0}
          max={30}
          value={form.mental_health_days}
          onChange={(value) => onFieldChange("mental_health_days", value)}
        />
        <SliderField
          label="Physical health days"
          min={0}
          max={30}
          value={form.physical_health_days}
          onChange={(value) => onFieldChange("physical_health_days", value)}
        />
      </InputSection>

      <InputSection title="Lifestyle and Access" icon={<Activity className="h-4 w-4" />} defaultOpen={false}>
        <ToggleGrid>
          <Toggle
            label="Physical activity"
            checked={form.phys_activity}
            onChange={(value) => onFieldChange("phys_activity", value)}
          />
          <Toggle
            label="Fruit intake"
            checked={form.fruits}
            onChange={(value) => onFieldChange("fruits", value)}
          />
          <Toggle
            label="Vegetable intake"
            checked={form.veggies}
            onChange={(value) => onFieldChange("veggies", value)}
          />
          <Toggle
            label="Heavy alcohol"
            checked={form.heavy_alcohol_consump}
            onChange={(value) => onFieldChange("heavy_alcohol_consump", value)}
          />
          <Toggle
            label="Healthcare coverage"
            checked={form.any_healthcare}
            onChange={(value) => onFieldChange("any_healthcare", value)}
          />
          <Toggle
            label="Cost barrier"
            checked={form.no_doc_bc_cost}
            onChange={(value) => onFieldChange("no_doc_bc_cost", value)}
          />
        </ToggleGrid>
        <SliderField
          label="Education level"
          min={1}
          max={6}
          value={form.education}
          onChange={(value) => onFieldChange("education", value)}
        />
        <SliderField
          label="Income level"
          min={1}
          max={8}
          value={form.income}
          onChange={(value) => onFieldChange("income", value)}
        />
      </InputSection>

      <button
        type="button"
        onClick={onSubmit}
        disabled={isPending}
        className="mt-4 flex h-12 w-full items-center justify-center gap-2 rounded-md bg-info px-4 text-sm font-bold text-canvas transition hover:bg-[#7bbce9] disabled:cursor-not-allowed disabled:opacity-60"
      >
        {isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
        Analyze Patient
      </button>

      {error ? (
        <div className="mt-4 rounded-md border border-danger/40 bg-danger/10 p-3 text-sm text-[#ffb4ad]">
          {error.message}
        </div>
      ) : null}
    </aside>
  );
}
