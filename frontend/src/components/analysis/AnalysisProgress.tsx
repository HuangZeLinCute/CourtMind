import { Loader2 } from "lucide-react"

import { AnalysisStages } from "@/components/analysis/AnalysisStages"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { useLanguage } from "@/i18n/LanguageProvider"
import type { JobStatusResponse } from "@/types"

interface AnalysisProgressProps {
  status: JobStatusResponse
}

export function AnalysisProgress({ status }: AnalysisProgressProps) {
  const { t } = useLanguage()
  const failed = status.status === "failed"
  const completed = status.status === "completed"

  return (
    <div className="space-y-5 rounded-lg border bg-card p-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {!completed && !failed && <Loader2 className="h-4 w-4 animate-spin text-primary" />}
          <h3 className="text-sm font-semibold">
            {failed ? t("analysis.failed") : completed ? t("analysis.completed") : t("analysis.analyzingMatch")}
          </h3>
        </div>
        <Badge variant={failed ? "destructive" : completed ? "success" : "secondary"}>
          {status.progress}%
        </Badge>
      </div>

      <Progress value={status.progress} className="h-2" />

      <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
        <span className="text-muted-foreground">
          {status.current_frame} / {status.total_frames} {t("analysis.frames")}
        </span>
        <span className="font-medium">{status.stage ?? "Processing"}</span>
      </div>

      <Separator />

      <AnalysisStages status={status} />

      {failed && status.error && (
        <p className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {status.error}
        </p>
      )}
    </div>
  )
}
