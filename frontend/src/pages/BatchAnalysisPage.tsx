import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { CheckCircle2, Files, Loader2, Play, Trash2, UploadCloud, XCircle } from "lucide-react"
import { Link } from "react-router-dom"
import { toast } from "sonner"

import { AnalysisConfiguration } from "@/components/analysis/AnalysisConfiguration"
import { PageHeader } from "@/components/layout/PageHeader"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { useLanguage } from "@/i18n/LanguageProvider"
import { api } from "@/lib/api"
import { DEFAULT_OPTIONS } from "@/lib/defaults"
import type { AnalysisOptions, BatchJobStatus, BatchStatusResponse, VideoInfo } from "@/types"

const MAX_BATCH_SIZE = 50
const TERMINAL_BATCH = new Set(["completed", "completed_with_errors"])

function fileKey(file: File) {
  return `${file.name}:${file.size}:${file.lastModified}`
}

function errorMessage(error: unknown, fallback: string) {
  return (
    (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
    (error instanceof Error ? error.message : fallback)
  )
}

export function BatchAnalysisPage() {
  const { t } = useLanguage()
  const inputRef = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<File[]>([])
  const [videos, setVideos] = useState<VideoInfo[]>([])
  const [options, setOptions] = useState<AnalysisOptions>({ ...DEFAULT_OPTIONS })
  const [uploading, setUploading] = useState(false)
  const [uploadIndex, setUploadIndex] = useState(0)
  const [starting, setStarting] = useState(false)
  const [batch, setBatch] = useState<BatchStatusResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const addFiles = useCallback((incoming: FileList | File[]) => {
    setFiles((current) => {
      const merged = new Map(current.map((file) => [fileKey(file), file]))
      Array.from(incoming).forEach((file) => {
        if (file.type.startsWith("video/") || /\.(mp4|mov|avi|mkv)$/i.test(file.name)) {
          merged.set(fileKey(file), file)
        }
      })
      return Array.from(merged.values()).slice(0, MAX_BATCH_SIZE)
    })
    setVideos([])
    setBatch(null)
    setError(null)
  }, [])

  const uploadAll = useCallback(async () => {
    if (!files.length) return
    setUploading(true)
    setUploadIndex(0)
    setError(null)
    const uploaded: VideoInfo[] = []
    try {
      for (let index = 0; index < files.length; index += 1) {
        setUploadIndex(index + 1)
        const form = new FormData()
        form.append("file", files[index])
        const { data } = await api.post<VideoInfo>("/videos", form, {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 300_000,
        })
        uploaded.push(data)
        setVideos([...uploaded])
      }
      toast.success(`${uploaded.length} ${t("batch.uploaded")}`)
    } catch (uploadError) {
      setError(errorMessage(uploadError, "Batch upload failed"))
    } finally {
      setUploading(false)
    }
  }, [files, t])

  const startBatch = useCallback(async () => {
    if (!videos.length || videos.length !== files.length) return
    setStarting(true)
    setError(null)
    try {
      const { data } = await api.post<BatchStatusResponse>("/batches", {
        video_ids: videos.map((video) => video.video_id),
        options,
      })
      setBatch(data)
      toast.success(t("analysis.analyzingMatch"))
    } catch (startError) {
      setError(errorMessage(startError, "Could not start batch"))
    } finally {
      setStarting(false)
    }
  }, [files.length, options, t, videos])

  useEffect(() => {
    if (!batch || TERMINAL_BATCH.has(batch.status)) return
    let cancelled = false
    const poll = async () => {
      try {
        const { data } = await api.get<BatchStatusResponse>(`/batches/${batch.batch_id}`)
        if (!cancelled) setBatch(data)
      } catch (pollError) {
        if (!cancelled) setError(errorMessage(pollError, "Could not load batch status"))
      }
    }
    const timer = window.setInterval(() => void poll(), 1000)
    void poll()
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [batch?.batch_id, batch?.status])

  const totalBytes = useMemo(() => files.reduce((sum, file) => sum + file.size, 0), [files])
  const reset = () => {
    setFiles([])
    setVideos([])
    setBatch(null)
    setError(null)
    setUploadIndex(0)
  }

  return (
    <div className="mx-auto max-w-6xl space-y-6">
      <PageHeader title={t("batch.title")} description={t("batch.description")} />

      {!batch && (
        <>
          <Card>
            <CardHeader className="flex-row items-center justify-between space-y-0">
              <CardTitle className="text-base">{t("batch.select")}</CardTitle>
              {files.length > 0 && (
                <Button variant="ghost" size="sm" onClick={reset} disabled={uploading}>
                  <Trash2 className="mr-2 h-4 w-4" />
                  {t("batch.clear")}
                </Button>
              )}
            </CardHeader>
            <CardContent className="space-y-4">
              <div
                role="button"
                tabIndex={0}
                onClick={() => !uploading && inputRef.current?.click()}
                onKeyDown={(event) => event.key === "Enter" && inputRef.current?.click()}
                onDragOver={(event) => event.preventDefault()}
                onDrop={(event) => {
                  event.preventDefault()
                  if (!uploading) addFiles(event.dataTransfer.files)
                }}
                className="flex cursor-pointer flex-col items-center gap-3 rounded-lg border border-dashed p-8 text-center hover:bg-accent/50"
              >
                <UploadCloud className="h-8 w-8 text-primary" />
                <div>
                  <p className="text-sm font-medium">{t("batch.drop")}</p>
                  <p className="mt-1 text-xs text-muted-foreground">{t("batch.hint")}</p>
                </div>
              </div>
              <input
                ref={inputRef}
                type="file"
                multiple
                accept="video/*,.mp4,.mov,.avi,.mkv"
                className="hidden"
                onChange={(event) => {
                  if (event.target.files) addFiles(event.target.files)
                  event.target.value = ""
                }}
              />

              {files.length > 0 && (
                <div className="space-y-3">
                  <div className="flex items-center justify-between text-sm">
                    <span>{t("batch.selected")}: {files.length}</span>
                    <span className="text-muted-foreground">{(totalBytes / 1024 / 1024).toFixed(1)} MB</span>
                  </div>
                  <div className="max-h-52 divide-y overflow-y-auto rounded-md border">
                    {files.map((file, index) => (
                      <div key={fileKey(file)} className="flex items-center gap-3 px-3 py-2 text-sm">
                        <Files className="h-4 w-4 shrink-0 text-muted-foreground" />
                        <span className="min-w-0 flex-1 truncate">{index + 1}. {file.name}</span>
                        {videos[index] && <CheckCircle2 className="h-4 w-4 text-emerald-500" />}
                      </div>
                    ))}
                  </div>
                  {uploading && (
                    <div className="space-y-2">
                      <Progress value={(uploadIndex / files.length) * 100} />
                      <p className="text-xs text-muted-foreground">
                        {t("batch.uploading")} {uploadIndex} / {files.length}
                      </p>
                    </div>
                  )}
                  {videos.length !== files.length && (
                    <Button onClick={() => void uploadAll()} disabled={uploading}>
                      {uploading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                      {t("batch.upload")}
                    </Button>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {videos.length === files.length && videos.length > 0 && (
            <Card>
              <CardHeader><CardTitle className="text-base">{t("batch.configuration")}</CardTitle></CardHeader>
              <CardContent className="space-y-5">
                <AnalysisConfiguration value={options} onChange={setOptions} />
                <p className="text-xs text-muted-foreground">{t("batch.autoCourt")}</p>
                <Button className="w-full" onClick={() => void startBatch()} disabled={starting}>
                  {starting ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Play className="mr-2 h-4 w-4" />}
                  {starting ? t("batch.starting") : t("batch.start")}
                </Button>
              </CardContent>
            </Card>
          )}
        </>
      )}

      {error && <p className="rounded-md bg-destructive/10 px-3 py-2 text-sm text-destructive">{error}</p>}

      {batch && <BatchQueue batch={batch} />}
    </div>
  )
}

function BatchQueue({ batch }: { batch: BatchStatusResponse }) {
  const { t } = useLanguage()
  return (
    <Card>
      <CardHeader className="space-y-3">
        <div className="flex items-center justify-between gap-3">
          <CardTitle className="text-base">{t("batch.queue")}</CardTitle>
          <Badge variant={batch.failed ? "warning" : TERMINAL_BATCH.has(batch.status) ? "success" : "secondary"}>
            {batch.finished} / {batch.total} {t("batch.summary")}
          </Badge>
        </div>
        <Progress value={batch.progress} />
        <p className="text-xs text-muted-foreground">
          {batch.status === "completed_with_errors" ? t("batch.completedWithErrors") :
            batch.status === "completed" ? t("batch.allCompleted") : `${batch.progress}%`}
        </p>
      </CardHeader>
      <CardContent className="space-y-3">
        {batch.jobs.map((job, index) => <BatchJobRow key={job.job_id} job={job} index={index} />)}
      </CardContent>
    </Card>
  )
}

function BatchJobRow({ job, index }: { job: BatchJobStatus; index: number }) {
  const { t } = useLanguage()
  const failed = job.status === "failed"
  const completed = job.status === "completed"
  return (
    <div className="space-y-2 rounded-md border p-3">
      <div className="flex flex-wrap items-center gap-2">
        {failed ? <XCircle className="h-4 w-4 text-destructive" /> :
          completed ? <CheckCircle2 className="h-4 w-4 text-emerald-500" /> :
            <Loader2 className={`h-4 w-4 ${job.status === "processing" ? "animate-spin text-primary" : "text-muted-foreground"}`} />}
        <span className="min-w-0 flex-1 truncate text-sm font-medium">{index + 1}. {job.filename}</span>
        <Badge variant={failed ? "destructive" : completed ? "success" : "outline"}>
          {t(`common.${job.status}`)}
        </Badge>
        {(completed || failed) && (
          <Button asChild variant="outline" size="sm">
            <Link to={`/analysis/${job.job_id}`}>{t("batch.openResult")}</Link>
          </Button>
        )}
      </div>
      <Progress value={job.progress} />
      <p className={`text-xs ${failed ? "text-destructive" : "text-muted-foreground"}`}>
        {job.error ?? job.stage ?? `${job.progress}%`}
      </p>
    </div>
  )
}

