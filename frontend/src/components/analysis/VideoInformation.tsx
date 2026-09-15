import { Film } from "lucide-react"

import { formatBytes, formatDuration } from "@/lib/utils"
import { useLanguage } from "@/i18n/LanguageProvider"
import type { VideoInfo } from "@/types"

interface VideoInformationProps {
  video: VideoInfo
}

export function VideoInformation({ video }: VideoInformationProps) {
  const { t } = useLanguage()
  const rows: Array<[string, string]> = [
    [t("videoInfo.filename"), video.filename],
    [t("videoInfo.resolution"), `${video.width} × ${video.height}`],
    [t("videoInfo.fps"), `${video.fps}`],
    [t("videoInfo.duration"), formatDuration(video.duration)],
    [t("videoInfo.fileSize"), formatBytes(video.file_size_bytes)],
  ]
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-2 rounded-lg border bg-card p-4 sm:grid-cols-3 lg:grid-cols-5">
      {rows.map(([k, v]) => (
        <div key={k} className="min-w-0">
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Film className="h-3 w-3" />
            {k}
          </div>
          <div className="mt-0.5 truncate text-sm font-medium" title={v}>
            {v}
          </div>
        </div>
      ))}
    </div>
  )
}
