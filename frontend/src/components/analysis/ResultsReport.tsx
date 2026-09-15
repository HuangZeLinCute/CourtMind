import { Activity, BarChart3, Clock3, Eye, Gauge, Target } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useLanguage } from "@/i18n/LanguageProvider"
import { toFullUrl } from "@/lib/api"
import { formatDuration } from "@/lib/utils"
import type { JobResult } from "@/types"

export function ResultsReport({ result }: { result: JobResult }) {
  const { t } = useLanguage()
  const report = result.report
  if (!report) {
    return <div className="rounded-lg border bg-card p-6 text-sm text-muted-foreground">{t("report.unavailable")}</div>
  }

  const summary = report.summary
  const quality = report.data_quality
  const videoUrl = toFullUrl(result.video_url)
  const cards = [
    { label: t("report.rallies"), value: summary.rally_count, icon: Activity },
    { label: t("report.hits"), value: summary.total_hits, icon: Target },
    { label: t("report.avgHits"), value: summary.average_hits_per_rally, icon: BarChart3 },
    { label: t("report.longest"), value: `${summary.longest_rally_hits} ${t("report.shots")}`, icon: Gauge },
    { label: t("report.activeTime"), value: formatDuration(summary.active_rally_time_sec), icon: Clock3 },
    { label: t("report.coverage"), value: `${Math.round(quality.shuttle_coverage * 100)}%`, icon: Eye },
  ]

  return (
    <div className="space-y-5">
      <div className="rounded-lg border bg-gradient-to-br from-primary/10 via-card to-card p-5">
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="text-lg font-semibold">{report.title}</h3>
          <Badge variant={quality.shuttle_coverage >= 0.6 ? "success" : "warning"}>
            {t("report.confidence")} {Math.round(quality.average_hit_confidence * 100)}%
          </Badge>
        </div>
        <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{report.overview}</p>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {cards.map(({ label, value, icon: Icon }) => (
          <Card key={label}>
            <CardContent className="flex items-center gap-3 p-4">
              <div className="rounded-md bg-primary/10 p-2"><Icon className="h-4 w-4 text-primary" /></div>
              <div><p className="text-xs text-muted-foreground">{label}</p><p className="text-lg font-semibold">{value}</p></div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        {report.insights.map((insight, index) => (
          <Card key={`${insight.kind}-${index}`}>
            <CardHeader className="p-4 pb-2"><CardTitle className="text-sm">{insight.title}</CardTitle></CardHeader>
            <CardContent className="p-4 pt-0 text-sm leading-relaxed text-muted-foreground">{insight.message}</CardContent>
          </Card>
        ))}
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {(["upper", "lower"] as const).map((side) => {
          const player = report.players[side]
          return (
            <Card key={side}>
              <CardHeader className="p-4 pb-2"><CardTitle className="text-sm">{t(`report.player.${side}`)}</CardTitle></CardHeader>
              <CardContent className="grid grid-cols-3 gap-3 p-4 pt-0">
                <Metric label={t("report.playerHits")} value={String(player.hit_count)} />
                <Metric label={t("report.distance")} value={`${player.distance_m.toFixed(1)}m`} />
                <Metric label={t("report.maxSpeed")} value={`${player.max_speed_mps.toFixed(1)}m/s`} />
              </CardContent>
            </Card>
          )
        })}
      </div>

      <Card>
        <CardHeader><CardTitle className="text-base">{t("report.rallyTimeline")}</CardTitle></CardHeader>
        <CardContent className="space-y-2">
          {report.rallies.length === 0 && <p className="text-sm text-muted-foreground">{t("report.noRallies")}</p>}
          {report.rallies.map((rally) => (
            <div key={rally.rally_id} className="grid items-center gap-3 rounded-md border p-3 sm:grid-cols-[70px_1fr_auto]">
              <Badge variant="outline">Rally {rally.rally_id}</Badge>
              <div className="min-w-0">
                <p className="text-sm font-medium">
                  {rally.hit_count} {t("report.shots")} · {rally.duration_sec.toFixed(1)}s · {t(`report.intensity.${rally.intensity}`)}
                </p>
                <p className="text-xs text-muted-foreground">
                  {rally.start_sec.toFixed(1)}–{rally.end_sec.toFixed(1)}s · {t("report.upper")} {rally.hits_by_player.upper} / {t("report.lower")} {rally.hits_by_player.lower} · {t("report.confidence")} {Math.round(rally.confidence * 100)}%
                </p>
              </div>
              {videoUrl && (
                <Button asChild variant="outline" size="sm">
                  <a href={`${videoUrl}#t=${rally.start_sec},${rally.end_sec}`} target="_blank" rel="noreferrer">{t("report.watch")}</a>
                </Button>
              )}
            </div>
          ))}
        </CardContent>
      </Card>

      <p className="rounded-md bg-muted/60 p-3 text-xs leading-relaxed text-muted-foreground">{report.limitations}</p>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div><p className="text-[11px] text-muted-foreground">{label}</p><p className="mt-1 text-sm font-semibold">{value}</p></div>
}
