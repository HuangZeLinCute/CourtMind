import { toFullUrl } from "@/lib/api"

interface VideoPreviewProps {
  url: string | null | undefined
  className?: string
}

export function VideoPreview({ url, className }: VideoPreviewProps) {
  const full = toFullUrl(url)
  if (!full) return null
  return (
    <video
      src={full}
      controls
      preload="metadata"
      className={className ?? "max-h-[420px] w-full rounded-lg border bg-black"}
    />
  )
}
