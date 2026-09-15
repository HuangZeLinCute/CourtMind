import { useEffect, useMemo, useRef, useState } from "react"
import { Activity, CircleDot, Download, Gauge, Loader2, Route } from "lucide-react"

import { api, toFullUrl } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { useLanguage } from "@/i18n/LanguageProvider"
import type { JobResult, JobTimeline, TimelinePlayer, TimelinePoint } from "@/types"

function PlayerPanel({ label, player, color, zh }: { label: string; player?: TimelinePlayer; color: string; zh: boolean }) {
  return <div className="rounded-xl border bg-background p-3.5">
    <div className="mb-3 flex items-center gap-2 text-xs font-medium"><span className={`h-2 w-2 rounded-full ${color}`} />{label}</div>
    <div className="mb-3"><p className="text-[10px] text-muted-foreground">{zh ? "当前速度" : "Current speed"}</p><p className="mt-1 text-xl font-semibold tabular-nums">{player?.speed_mps?.toFixed(1) ?? "—"}<span className="ml-1 text-[10px] font-normal text-muted-foreground">m/s</span></p></div>
    <div className="grid grid-cols-[1fr_auto_auto] gap-x-3 gap-y-1.5 border-t pt-2 text-[10px]">
      <span className="text-muted-foreground">{zh ? "本回合" : "Rally"}</span><span className="tabular-nums">{player?.rally_distance_m?.toFixed(1) ?? "—"} m</span><span className="tabular-nums text-muted-foreground">{zh ? "均" : "avg"} {player?.rally_average_speed_mps?.toFixed(1) ?? "—"} / {zh ? "峰" : "max"} {player?.rally_max_speed_mps?.toFixed(1) ?? "—"}</span>
      <span className="text-muted-foreground">{zh ? "全场" : "Match"}</span><span className="tabular-nums">{player?.match_distance_m?.toFixed(1) ?? "—"} m</span><span className="tabular-nums text-muted-foreground">{zh ? "均" : "avg"} {player?.match_average_speed_mps?.toFixed(1) ?? "—"} / {zh ? "峰" : "max"} {player?.match_max_speed_mps?.toFixed(1) ?? "—"}</span>
    </div>
  </div>
}

export function ResultsVideo({ result }: { result: JobResult }) {
  const { t, language } = useLanguage()
  const zh = language === "zh"
  const videoUrl = toFullUrl(result.video_url)
  const videoRef = useRef<HTMLVideoElement>(null)
  const [timeline, setTimeline] = useState<JobTimeline | null>(null)
  const [currentTime, setCurrentTime] = useState(0)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    void api.get<JobTimeline>(`/jobs/${result.job_id}/timeline`, { signal: controller.signal })
      .then(({ data }) => { if (!controller.signal.aborted) setTimeline(data) })
      .catch(() => { if (!controller.signal.aborted) setTimeline(null) })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [result.job_id])

  const point = useMemo<TimelinePoint | undefined>(() => {
    const points = timeline?.points
    if (!points?.length) return undefined
    let low = 0; let high = points.length - 1
    while (low < high) {
      const mid = Math.ceil((low + high) / 2)
      if (points[mid].time_sec <= currentTime) low = mid
      else high = mid - 1
    }
    return points[low]
  }, [timeline, currentTime])

  if (!videoUrl) {
    return (
      <div className="rounded-lg border bg-card p-6 text-sm text-muted-foreground">
        {t("results.videoNotFound")}
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="grid min-h-0 overflow-hidden rounded-2xl border bg-muted/20 xl:grid-cols-[minmax(0,1fr)_320px]">
        <div className="flex min-h-[320px] items-center bg-black">
          <video ref={videoRef} src={videoUrl} controls className="max-h-[72vh] w-full" preload="metadata" onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)} onSeeked={(event) => setCurrentTime(event.currentTarget.currentTime)} />
        </div>
        <aside className="border-t p-4 xl:border-l xl:border-t-0">
          <div className="mb-4 flex items-center justify-between">
            <div><p className="text-sm font-semibold">{zh ? "实时比赛数据" : "Live match data"}</p><p className="mt-0.5 text-[11px] text-muted-foreground">{zh ? "与视频时间轴同步" : "Synced to video playback"}</p></div>
            {loading ? <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" /> : <span className={`h-2 w-2 rounded-full ${point ? "bg-emerald-500" : "bg-muted-foreground/40"}`} />}
          </div>
          <div className="mb-3 grid grid-cols-3 gap-2">
            {[
              { icon: Activity, value: point?.rally_id ? `#${point.rally_id}` : "—", label: zh ? "当前回合" : "Rally" },
              { icon: CircleDot, value: point?.rally_id ? `${point.rally_hit_count}/${point.rally_total_hits}` : "—", label: zh ? "击球进度" : "Hits" },
              { icon: Gauge, value: point?.shuttlecock.visible ? (zh ? "已检测" : "Seen") : (zh ? "未检测" : "Lost"), label: zh ? "羽毛球" : "Shuttle" },
            ].map((item) => <div key={item.label} className="rounded-xl border bg-background px-2 py-3 text-center"><item.icon className="mx-auto mb-1.5 h-3.5 w-3.5 text-primary" /><p className="truncate text-sm font-semibold tabular-nums">{item.value}</p><p className="mt-0.5 text-[9px] text-muted-foreground">{item.label}</p></div>)}
          </div>
          <div className="space-y-3">
            <PlayerPanel label={zh ? "上半场球员" : "Upper player"} player={point?.players.upper} color="bg-cyan-500" zh={zh} />
            <PlayerPanel label={zh ? "下半场球员" : "Lower player"} player={point?.players.lower} color="bg-amber-500" zh={zh} />
          </div>
          <div className="mt-3 flex items-center justify-between rounded-xl bg-primary/[0.06] px-3 py-2.5 text-[11px] text-muted-foreground"><span className="flex items-center gap-1.5"><Route className="h-3.5 w-3.5 text-primary" />{zh ? "视频位置" : "Position"}</span><span className="font-medium tabular-nums text-foreground">{Math.floor(currentTime / 60)}:{String(Math.floor(currentTime % 60)).padStart(2, "0")} · F{point?.frame ?? 0}</span></div>
        </aside>
      </div>
      <div className="flex items-center justify-end">
        <Button variant="outline" size="sm" asChild>
          <a href={videoUrl} download>
            <Download className="h-4 w-4" /> {t("results.downloadVideo")}
          </a>
        </Button>
      </div>
    </div>
  )
}
