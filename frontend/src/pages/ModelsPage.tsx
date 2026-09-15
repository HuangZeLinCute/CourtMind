import { useSystemStatus } from "@/hooks/useSystemStatus"
import { useLanguage } from "@/i18n/LanguageProvider"
import { PageHeader } from "@/components/layout/PageHeader"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"

export function ModelsPage({ embedded = false }: { embedded?: boolean }) {
  const { status, loading, error } = useSystemStatus()
  const { t } = useLanguage()

  return (
    <div className="space-y-6">
      {!embedded && <PageHeader
        title={t("models.title")}
        description={t("models.description")}
      />}

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {loading || !status ? (
        <div className="grid gap-3 sm:grid-cols-2">
          {Array.from({ length: 4 }).map((_, i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <Card>
            <CardHeader className="p-4 pb-1">
              <CardTitle className="flex items-center justify-between text-sm">
                {t("models.device")}
                <Badge variant={status.gpu_available ? "success" : "secondary"}>
                  {status.device}
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 pt-1 text-xs text-muted-foreground">
              {status.gpu_available ? status.gpu_name ?? "GPU" : "CPU (torch)"}
            </CardContent>
          </Card>

          <ModelCard
            title="CourtKeyNet"
            ready={status.courtkeynet_ready}
            path={status.courtkeynet_weights}
          />
          <ModelCard
            title="Shuttlecock (YOLO)"
            ready={status.ball_model_ready}
            path={status.ball_model}
          />
          <ModelCard
            title="TrackNetV3 Tracker"
            ready={!!status.tracknet_tracker_ready}
            path={status.tracknet_tracker_weights ?? "weights/tracknet/tracknet_v3_tracker.pt"}
          />
          <ModelCard
            title="TrackNetV3 Rectifier"
            ready={!!status.tracknet_rectifier_ready}
            path={status.tracknet_rectifier_weights ?? "weights/tracknet/tracknet_v3_rectifier.pt"}
          />
          <ModelCard
            title="TrackNetV4 Type B"
            ready={!!(
              status.tracknet_v4_weights_ready && status.tracknet_v4_dependency_ready
            )}
            path={status.tracknet_v4_weights ?? "weights/tracknet_v4/tracknet_v4_type_b.keras"}
          />
          <ModelCard
            title="YOLO Pose"
            ready={status.yolo_pose_model_ready}
            path={status.yolo_pose_model}
          />
          <ModelCard title="RTMPose" ready={status.rtmpose_model_ready} path="weights/rtmpose-s…onnx" />
          <ModelCard title="RTMO" ready={status.rtmo_model_ready} path="weights/rtmo-s…onnx" />
        </div>
      )}
    </div>
  )
}

function ModelCard({ title, ready, path }: { title: string; ready: boolean; path: string }) {
  const { t } = useLanguage()
  return (
    <Card>
      <CardHeader className="p-4 pb-1">
        <CardTitle className="flex items-center justify-between text-sm">
          {title}
          <Badge variant={ready ? "success" : "destructive"}>
            {ready ? t("models.ready") : t("models.missing")}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="p-4 pt-1">
        <p className="truncate font-mono text-xs text-muted-foreground" title={path}>
          {path}
        </p>
      </CardContent>
    </Card>
  )
}
