import { formatDuration } from "@/lib/utils"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { useLanguage } from "@/i18n/LanguageProvider"
import type { JobResult } from "@/types"

interface OverviewDatum {
  label: string
  value: string
}

function MetadataView({ metadata }: { metadata: Record<string, unknown> }) {
  const { t } = useLanguage()
  const video = (metadata.video ?? {}) as Record<string, unknown>
  const court = (metadata.court ?? {}) as Record<string, unknown>
  const shuttleRaw = (metadata.models as Record<string, unknown> | undefined)?.shuttlecock
  const shuttleLabel =
    shuttleRaw === "tracknet_v4_tracknet_v3_yolo11_ensemble"
      ? "TrackNetV4 + TrackNetV3 + YOLO11 Ensemble"
      : shuttleRaw === "tracknet_v3"
      ? "TrackNetV3"
      : shuttleRaw === "tracknet_v4_type_b"
        ? "TrackNetV4 Type B"
        : String(shuttleRaw ?? "—")

  const rows: OverviewDatum[] = [
    { label: t("results.matchDuration"), value: formatDuration(Number(video.duration_sec ?? 0)) },
    { label: t("results.frames"), value: String(video.total_frames ?? "—") },
    { label: t("results.fps"), value: String(video.fps ?? "—") },
    { label: t("results.resolution"), value: `${video.width ?? "?"} × ${video.height ?? "?"}` },
    {
      label: t("results.courtDetector"),
      value: String(court.detector ?? "—"),
    },
    {
      label: t("results.shuttlecockModel"),
      value: shuttleLabel,
    },
    {
      label: t("results.coordinateSystem"),
      value: `${(court.coordinate_system as Record<string, unknown> | undefined)?.width ?? "?"} × ${(court.coordinate_system as Record<string, unknown> | undefined)?.length ?? "?"} m`,
    },
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
            <p className="truncate text-base font-semibold" title={row.value}>
              {row.value}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}

export function ResultsOverview({ result }: { result: JobResult }) {
  return <MetadataView metadata={result.metadata ?? {}} />
}
