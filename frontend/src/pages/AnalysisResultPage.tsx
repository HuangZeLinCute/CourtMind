import { useEffect, useState } from "react"
import { useParams } from "react-router-dom"
import { Loader2 } from "lucide-react"

import { AnalysisProgress } from "@/components/analysis/AnalysisProgress"
import { ResultsWorkspace } from "@/components/analysis/ResultsWorkspace"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { PageHeader } from "@/components/layout/PageHeader"
import { useAnalysis } from "@/hooks/useAnalysis"
import { useJobPolling } from "@/hooks/useJobPolling"
import { useLanguage } from "@/i18n/LanguageProvider"

/**
 * /analysis/:jobId — deep link to a completed (or still running) analysis.
 */
export function AnalysisResultPage() {
  const { t } = useLanguage()
  const { jobId = "" } = useParams<{ jobId: string }>()
  const { result, error: resultError, loadResult } = useAnalysis(null)
  const [error, setError] = useState<string | null>(null)
  const { status: pollStatus } = useJobPolling(jobId, true)

  const jobDone = pollStatus?.status === "completed" || pollStatus?.status === "failed"

  useEffect(() => {
    if (jobDone && !result) {
      void loadResult(jobId)
    }
  }, [jobDone, jobId, loadResult, result])

  useEffect(() => {
    if (pollStatus?.status === "failed") {
      setError(pollStatus.error ?? "Analysis failed")
    }
  }, [pollStatus])

  useEffect(() => {
    if (resultError) setError(resultError)
  }, [resultError])

  const running =
    pollStatus && (pollStatus.status === "queued" || pollStatus.status === "processing")

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <PageHeader title={t("results.title")} actions={<Badge variant="secondary">ID: {jobId}</Badge>} />

      {error && !result && (
        <div className="space-y-3">
          <div className="rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
          <Button variant="outline" size="sm" onClick={() => void loadResult(jobId)}>
            {t("results.retryLoading")}
          </Button>
        </div>
      )}

      {!pollStatus && !error && (
        <div className="flex items-center justify-center gap-2 py-12 text-sm text-muted-foreground">
          <Loader2 className="h-4 w-4 animate-spin" /> {t("common.loading")}
        </div>
      )}

      {running && pollStatus && <AnalysisProgress status={pollStatus} />}

      {result && <ResultsWorkspace result={result} />}

      {jobDone && !result && !error && (
        <div className="flex justify-center py-8">
          <Button onClick={() => void loadResult(jobId)}>{t("newAnalysis.loadResults")}</Button>
        </div>
      )}
    </div>
  )
}
