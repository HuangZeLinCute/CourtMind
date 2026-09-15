import { cn } from "@/lib/utils"
import { Check } from "lucide-react"

export interface StepperStep {
  id: string
  label: string
}

interface AnalysisStepperProps {
  steps: StepperStep[]
  current: number
  onStepClick?: (index: number) => void
}

export function AnalysisStepper({ steps, current, onStepClick }: AnalysisStepperProps) {
  return (
    <div className="flex items-center gap-2 overflow-x-auto pb-2">
      {steps.map((step, i) => {
        const state =
          i < current ? "done" : i === current ? "active" : "todo"
        return (
          <div key={step.id} className="flex shrink-0 items-center gap-2">
            <button
              type="button"
              disabled={!onStepClick || i > current}
              onClick={() => onStepClick?.(i)}
              className={cn(
                "flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm font-medium transition-colors",
                state === "active" &&
                  "border-primary bg-primary/10 text-primary",
                state === "done" &&
                  "border-emerald-500/40 bg-emerald-500/10 text-emerald-500",
                state === "todo" && "border-border text-muted-foreground",
                onStepClick && i <= current && "cursor-pointer hover:bg-accent",
              )}
            >
              {state === "done" ? (
                <Check className="h-3.5 w-3.5" />
              ) : (
                <span
                  className={cn(
                    "flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-bold",
                    state === "active" ? "bg-primary text-primary-foreground" : "bg-muted",
                  )}
                >
                  {i + 1}
                </span>
              )}
              {step.label}
            </button>
            {i < steps.length - 1 && (
              <div className="h-px w-6 bg-border" aria-hidden />
            )}
          </div>
        )
      })}
    </div>
  )
}
