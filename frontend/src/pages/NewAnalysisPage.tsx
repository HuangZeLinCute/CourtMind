import { useCallback, useState } from "react"
import { Link, useNavigate } from "react-router-dom"
import { toast } from "sonner"

import { AnalysisConfiguration } from "@/components/analysis/AnalysisConfiguration"
import { AnalysisProgress } from "@/components/analysis/AnalysisProgress"
import { AnalysisStepper } from "@/components/analysis/AnalysisStepper"
import { CourtCalibration } from "@/components/analysis/CourtCalibration"
import { ResultsWorkspace } from "@/components/analysis/ResultsWorkspace"
import { VideoInformation } from "@/components/analysis/VideoInformation"
import { VideoPreview } from "@/components/analysis/VideoPreview"
import { VideoUploader } from "@/components/analysis/VideoUploader"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useAnalysis } from "@/hooks/useAnalysis"
import { useJobPolling } from "@/hooks/useJobPolling"
import { useLanguage } from "@/i18n/LanguageProvider"
import { DEFAULT_OPTIONS } from "@/lib/defaults"
import type { AnalysisOptions, Corners, VideoInfo } from "@/types"

export function NewAnalysisPage() {
  const navigate = useNavigate()
  const { t } = useLanguage()
  const [step, setStep] = useState(0)
  const [video, setVideo] = useState<VideoInfo | null>(null)
  const [corners, setCorners] = useState<Corners | null>(null)
  const [detectorName, setDetectorName] = useState<string | null>(null)
  const [options, setOptions] = useState<AnalysisOptions>({ ...DEFAULT_OPTIONS })
  const [analysisName, setAnalysisName] = useState("")

  const { jobId, result, starting, error: analysisError, start, loadResult } = useAnalysis(
    video?.video_id ?? null,
  )
  const { status: jobStatus } = useJobPolling(jobId, step >= 3)

  const STEPS = [
    { id: "upload", label: t("newAnalysis.stepUpload") },
    { id: "court", label: t("newAnalysis.stepCourt") },
    { id: "configure", label: t("newAnalysis.stepConfigure") },
    { id: "analyze", label: t("newAnalysis.stepAnalyze") },
    { id: "results", label: t("newAnalysis.stepResults") },
  ]

  const onUploaded = useCallback(
    (v: VideoInfo) => {
      setVideo(v)
      setAnalysisName(v.filename.replace(/\.[^.]+$/, ""))
      setCorners(null)
      setStep(1)
      toast.success(t("upload.videoUploaded"))
    },
    [t],
  )

  const onCourtConfirmed = useCallback(
    (c: Corners, detector: string) => {
      setCorners(c)
      setDetectorName(detector)
      setStep(2)
      toast.success(t("court.confirm"))
    },
    [t],
  )

  const onStartAnalysis = useCallback(async () => {
    if (!video || !corners) {
      toast.error(t("newAnalysis.backToDashboard"))
      return
    }
    const id = await start(options, corners, analysisName)
    if (id) {
      setStep(3)
      toast.success(t("analysis.analyzingMatch"))
    }
  }, [video, corners, options, analysisName, start, t])

  // When the job completes, load results and advance.
  const pollDone = jobStatus?.status === "completed" || jobStatus?.status === "failed"
  const resultLoaded = result !== null

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold tracking-tight">{t("newAnalysis.title")}</h2>
        {step === 4 && result && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate(`/analysis/${result.job_id}`)}
          >
            {t("newAnalysis.openFullResult")}
          </Button>
        )}
      </div>

      <AnalysisStepper
        steps={STEPS}
        current={step}
        onStepClick={(i) => i < step && setStep(i)}
      />

      {step === 0 && <VideoUploader onUploaded={onUploaded} />}

      {step === 1 && video && (
        <div className="space-y-4">
          <VideoPreview url={video.url} className="max-h-[420px] w-full rounded-lg border bg-black" />
          <VideoInformation video={video} />
          <CourtCalibration video={video} onConfirmed={onCourtConfirmed} />
          <div className="flex justify-between">
            <Button variant="ghost" onClick={() => setStep(0)}>
              {t("newAnalysis.back")}
            </Button>
            <span className="text-xs text-muted-foreground">{t("newAnalysis.uploadHint")}</span>
          </div>
        </div>
      )}

      {step === 2 && video && corners && (
        <div className="space-y-5">
          <div className="rounded-lg border bg-card p-4">
            <Label htmlFor="analysis-name">{t("newAnalysis.name")}</Label>
            <Input
              id="analysis-name"
              className="mt-2"
              maxLength={120}
              value={analysisName}
              onChange={(event) => setAnalysisName(event.target.value)}
              placeholder={t("newAnalysis.namePlaceholder")}
            />
            <p className="mt-1 text-xs text-muted-foreground">{t("newAnalysis.nameHint")}</p>
          </div>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <div className="rounded-lg border bg-card p-4">
              <p className="text-xs font-medium text-muted-foreground">{t("configure.detectedCourt")}</p>
              <p className="mt-1 font-mono text-xs">
                {detectorName ?? "courtkeynet"} · 4 corners
              </p>
              <div className="mt-2 overflow-hidden rounded-md border bg-black">
                <VideoPreview
                  url={video.url}
                  className="pointer-events-none max-h-[200px] w-full"
                />
              </div>
            </div>
            <AnalysisConfiguration value={options} onChange={setOptions} />
          </div>

          {analysisError && (
            <p className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {analysisError}
            </p>
          )}

          <div className="flex justify-between">
            <Button variant="ghost" onClick={() => setStep(1)}>
              {t("newAnalysis.back")}
            </Button>
            <Button onClick={onStartAnalysis} disabled={starting}>
              {starting ? t("newAnalysis.starting") : t("newAnalysis.startAnalysis")}
            </Button>
          </div>
        </div>
      )}

      {step === 3 && jobStatus && (
        <div className="space-y-4">
          <AnalysisProgress status={jobStatus} />
          {pollDone && !resultLoaded && jobId && (
            <Button className="w-full" onClick={() => void loadResult(jobId)}>
              {t("newAnalysis.loadResults")}
            </Button>
          )}
          {pollDone && resultLoaded && (
            <Button
              className="w-full"
              onClick={() => {
                setStep(4)
                toast.success(t("newAnalysis.analysisCompleted"))
              }}
            >
              {t("newAnalysis.viewResults")}
            </Button>
          )}
          <p className="text-center text-xs text-muted-foreground">
            <Link to="/" className="hover:underline">
              {t("newAnalysis.backToDashboard")}
            </Link>
          </p>
        </div>
      )}

      {step === 4 && result && <ResultsWorkspace result={result} />}
    </div>
  )
}
