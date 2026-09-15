import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { useLanguage } from "@/i18n/LanguageProvider"

export interface VisualizationFlags {
  show_skeletons: boolean
  show_player_trajectories: boolean
  show_court_trajectory: boolean
  show_shuttlecock_trajectory: boolean
  show_player_stats: boolean
  show_pose_roi: boolean
}

interface VisualizationSettingsProps {
  value: VisualizationFlags
  onChange: (value: VisualizationFlags) => void
}

export function VisualizationSettings({ value, onChange }: VisualizationSettingsProps) {
  const { t } = useLanguage()

  const FLAG_LABELS: Array<{ key: keyof VisualizationFlags; label: string }> = [
    { key: "show_player_trajectories", label: t("configure.visPlayerTrajectory") },
    { key: "show_court_trajectory", label: t("configure.visCourtTrajectory") },
    { key: "show_shuttlecock_trajectory", label: t("configure.visShuttlecockTrajectory") },
  ]

  const toggle = (key: keyof VisualizationFlags) => {
    onChange({ ...value, [key]: !value[key] })
  }

  return (
    <div className="space-y-3">
      {FLAG_LABELS.map(({ key, label }) => (
        <div key={key} className="flex items-center justify-between">
          <Label htmlFor={`vis-${key}`} className="cursor-pointer text-sm font-normal">
            {label}
          </Label>
          <Switch
            id={`vis-${key}`}
            checked={value[key]}
            onCheckedChange={() => toggle(key)}
          />
        </div>
      ))}
      <p className="rounded-md bg-muted/60 px-3 py-2 text-xs leading-5 text-muted-foreground">
        {t("configure.visualizationDesc")}
      </p>
    </div>
  )
}
