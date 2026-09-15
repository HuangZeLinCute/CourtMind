import { Badge } from "@/components/ui/badge"
import type { JobStatusResponse } from "@/types"

export const ANALYSIS_STAGES = [
  "Queued",
  "Preparation",
  "Analysis",
  "Completed",
  "Failed",
] as const

interface AnalysisStagesProps {
  status: Pick<JobStatusResponse, "status" | "stage">
}

export function AnalysisStages({ status }: AnalysisStagesProps) {
  const current = (status.stage ?? "Analysis") as (typeof ANALYSIS_STAGES)[number]
  const failed = status.status === "failed"
  return (
    <div className="flex flex-wrap gap-2">
      {ANALYSIS_STAGES.map((stage) => {
        const active = current === stage
        const done = ANALYSIS_STAGES.indexOf(stage) < ANALYSIS_STAGES.indexOf(current)
        return (
          <Badge
            key={stage}
            variant={failed ? "destructive" : done ? "success" : active ? "default" : "outline"}
          >
            {stage}
          </Badge>
        )
      })}
    </div>
  )
}
