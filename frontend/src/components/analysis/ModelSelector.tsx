import { useSystemStatus } from "@/hooks/useSystemStatus"
import { useLanguage } from "@/i18n/LanguageProvider"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"

export interface ModelSelection {
  pose_family: "yolo-pose" | "rtmpose" | "rtmo"
  pose_mode: string
  shuttle_model: "yolo" | "tracknet" | "tracknet_v4" | "ensemble"
}

interface ModelSelectorProps {
  value: ModelSelection
  onChange: (value: ModelSelection) => void
}

export function ModelSelector({ value, onChange }: ModelSelectorProps) {
  const { status, loading } = useSystemStatus()
  const { t } = useLanguage()

  if (loading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
      </div>
    )
  }

  const rtmposeReady = !!status?.rtmpose_model_ready
  const rtmoReady = !!status?.rtmo_model_ready
  const tracknetReady = !!(status?.tracknet_tracker_ready && status?.tracknet_rectifier_ready)
  const tracknetV4Ready = !!(
    status?.tracknet_v4_weights_ready && status?.tracknet_v4_dependency_ready
  )
  const ensembleReady = !!(
    tracknetReady && tracknetV4Ready && status?.ball_model_ready
  )

  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <Label>{t("configure.shuttleModel")}</Label>
        <Select
          value={value.shuttle_model}
          onValueChange={(v) =>
            onChange({ ...value, shuttle_model: v as ModelSelection["shuttle_model"] })
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="ensemble" disabled={!ensembleReady}>
              {t("configure.shuttleEnsemble")} {!ensembleReady && "(unavailable)"}
            </SelectItem>
            <SelectItem value="tracknet_v4" disabled={!tracknetV4Ready}>
              {t("configure.shuttleTracknetV4")} {tracknetV4Ready ? "(ready)" : "(unavailable)"}
            </SelectItem>
            <SelectItem value="tracknet" disabled={!tracknetReady}>
              {t("configure.shuttleTracknet")}{" "}
              {tracknetReady ? "(ready)" : "(weights missing)"}
            </SelectItem>
            <SelectItem value="yolo">YOLO11 (yolo11s-ball.pt)</SelectItem>
          </SelectContent>
        </Select>
        {value.shuttle_model === "tracknet" && !tracknetReady && (
          <p className="text-xs text-destructive">{t("configure.tracknetUnavailable")}</p>
        )}
        {value.shuttle_model === "tracknet_v4" && !tracknetV4Ready && (
          <p className="text-xs text-destructive">{t("configure.tracknetV4Unavailable")}</p>
        )}
        {value.shuttle_model === "ensemble" && !ensembleReady && (
          <p className="text-xs text-destructive">{t("configure.ensembleUnavailable")}</p>
        )}
        {value.shuttle_model === "ensemble" && (
          <p className="text-xs text-muted-foreground">{t("configure.ensembleHint")}</p>
        )}
      </div>

      <div className="space-y-1.5">
        <Label>{t("configure.poseModel")}</Label>
        <Select
          value={value.pose_family}
          onValueChange={(v) =>
            onChange({ ...value, pose_family: v as ModelSelection["pose_family"] })
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select pose model" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="yolo-pose">YOLO Pose (yolo11n-pose.pt)</SelectItem>
            <SelectItem value="rtmpose" disabled={!rtmposeReady}>
              RTMPose {rtmposeReady ? "(rtmpose-s onnx)" : "(weights missing)"}
            </SelectItem>
            <SelectItem value="rtmo" disabled={!rtmoReady}>
              RTMO {rtmoReady ? "(rtmo-s onnx)" : "(weights missing)"}
            </SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-1.5">
        <Label>{t("configure.poseMode")}</Label>
        <Select
          value={value.pose_mode}
          onValueChange={(v) => onChange({ ...value, pose_mode: v })}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select pose mode" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="balanced">Balanced</SelectItem>
            <SelectItem value="fast">Fast</SelectItem>
            <SelectItem value="accurate">Accurate</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="rounded-md bg-muted/60 px-3 py-2.5 text-xs text-muted-foreground">
        <p className="font-medium text-foreground">{t("configure.shuttlecockModel")}</p>
        <p className="mt-0.5 font-mono">yolo11s-ball.pt {status?.ball_model_ready ? "· ready" : "· missing"}</p>
      </div>
    </div>
  )
}
