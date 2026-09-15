import { useCallback, useRef, useState } from "react"
import { UploadCloud } from "lucide-react"

import { api } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { useLanguage } from "@/i18n/LanguageProvider"
import type { VideoInfo } from "@/types"

interface VideoUploaderProps {
  onUploaded: (video: VideoInfo) => void
}

export function VideoUploader({ onUploaded }: VideoUploaderProps) {
  const { t } = useLanguage()
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const upload = useCallback(
    async (file: File) => {
      setUploading(true)
      setError(null)
      try {
        const form = new FormData()
        form.append("file", file)
        const { data } = await api.post<VideoInfo>("/videos", form, {
          headers: { "Content-Type": "multipart/form-data" },
          timeout: 300_000,
        })
        onUploaded(data)
      } catch (e) {
        const detail =
          (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        setError(detail ?? (e instanceof Error ? e.message : "Upload failed"))
      } finally {
        setUploading(false)
      }
    },
    [onUploaded],
  )

const loadDemo = useCallback(async () => {
  setUploading(true)
  setError(null)
  try {
    const base = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8000"
    const resp = await fetch(`${base}/videos/demo.mp4`)
    if (!resp.ok) throw new Error("Demo video unavailable on the backend")
    const blob = await resp.blob()
    const file = new File([blob], "demo.mp4", { type: "video/mp4" })
    await upload(file)
  } catch (e) {
    setError(e instanceof Error ? e.message : "Could not load demo video")
    setUploading(false)
  }
}, [upload])

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          const file = e.dataTransfer.files?.[0]
          if (file) void upload(file)
        }}
        className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border border-dashed p-12 text-center transition-colors hover:bg-accent/50 data-[dragging=true]:border-primary"
        data-dragging={dragging}
      >
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary/10">
          <UploadCloud className="h-6 w-6 text-primary" />
        </div>
        <div>
          <p className="text-sm font-medium">{t("upload.dragDrop")}</p>
          <p className="mt-1 text-xs text-muted-foreground">{t("upload.orBrowse")}</p>
        </div>
        {uploading && <p className="text-xs text-muted-foreground">{t("upload.uploading")}</p>}
      </div>
      <input
        ref={inputRef}
        type="file"
        accept="video/*,.mp4,.mov,.avi,.mkv"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) void upload(file)
          e.target.value = ""
        }}
      />
      {error && <p className="mt-3 text-sm text-destructive">{error}</p>}
      <div className="mt-4 flex items-center gap-2">
        <Button variant="ghost" size="sm" className="text-muted-foreground" onClick={loadDemo}>
          {t("upload.tryDemo")}
        </Button>
        <span className="text-xs text-muted-foreground">{t("upload.demoHint")}</span>
      </div>
    </div>
  )
}
