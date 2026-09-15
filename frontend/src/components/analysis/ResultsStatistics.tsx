import { formatDuration } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useLanguage } from "@/i18n/LanguageProvider"

interface StatsRow {
  label: string
  value: string
}

/**
 * Reads only real values from metadata.json — no invented KPIs.
 */
export function ResultsStatistics({ metadata }: { metadata: Record<string, unknown> }) {
  const { t } = useLanguage()
  const video = (metadata.video ?? {}) as Record<string, unknown>
  const court = (metadata.court ?? {}) as Record<string, unknown>

  const rows: StatsRow[] = [
    { label: t("results.matchDuration"), value: formatDuration(Number(video.duration_sec ?? 0)) },
    { label: t("results.frames"), value: String(video.total_frames ?? "—") },
    { label: t("results.fps"), value: String(video.fps ?? "—") },
    { label: t("results.resolution"), value: `${video.width ?? "?"} × ${video.height ?? "?"}` },
    { label: t("results.courtDetector"), value: String(court.detector ?? "—") },
    { label: t("results.midHeight"), value: court.mid_height != null ? String(court.mid_height) : "—" },
  ]

  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {rows.map((row) => (
        <Card key={row.label}>
          <CardHeader className="p-4 pb-1">
            <CardTitle className="text-xs font-medium text-muted-foreground">
              {row.label}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-4 pt-1">
            <p className="text-base font-semibold">{row.value}</p>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
