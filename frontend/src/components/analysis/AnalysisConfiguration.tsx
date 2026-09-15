import { useEffect, useState } from "react"

import { ModelSelector, type ModelSelection } from "@/components/analysis/ModelSelector"
import {
  VisualizationSettings,
  type VisualizationFlags,
} from "@/components/analysis/VisualizationSettings"
import { Separator } from "@/components/ui/separator"
import { useLanguage } from "@/i18n/LanguageProvider"
import type { AnalysisOptions } from "@/types"

interface AnalysisConfigurationProps {
  value: AnalysisOptions
  onChange: (value: AnalysisOptions) => void
}

/**
 * Step 3 — Configure. Split into Models / Visualization; normal users never
 * touch model paths or CUDA params here (those live in advanced/system pages).
 */
export function AnalysisConfiguration({ value, onChange }: AnalysisConfigurationProps) {
  const { t } = useLanguage()
  const [models, setModels] = useState<ModelSelection>({
    pose_family: value.pose_family as ModelSelection["pose_family"],
    pose_mode: value.pose_mode,
    shuttle_model: value.shuttle_model ?? "tracknet",
  })
  const [vis, setVis] = useState<VisualizationFlags>({
    show_skeletons: value.show_skeletons,
    show_player_trajectories: value.show_player_trajectories,
    show_court_trajectory: value.show_court_trajectory,
    show_shuttlecock_trajectory: value.show_shuttlecock_trajectory,
    show_player_stats: value.show_player_stats,
    show_pose_roi: value.show_pose_roi,
  })

  useEffect(() => {
    onChange({
      ...value,
      pose_family: models.pose_family,
      pose_mode: models.pose_mode,
      shuttle_model: models.shuttle_model,
      ...vis,
      show_skeletons: true,
      show_player_stats: false,
      show_pose_roi: false,
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [models, vis])

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <section className="space-y-3">
        <div>
          <h3 className="text-sm font-semibold">{t("configure.models")}</h3>
          <p className="text-xs text-muted-foreground">{t("configure.modelsDesc")}</p>
        </div>
        <ModelSelector value={models} onChange={setModels} />
      </section>

      <section className="space-y-3">
        <div>
          <h3 className="text-sm font-semibold">{t("configure.visualization")}</h3>
          <p className="text-xs text-muted-foreground">{t("configure.visualizationDesc")}</p>
        </div>
        <div className="rounded-lg border bg-card p-4">
          <VisualizationSettings value={vis} onChange={setVis} />
        </div>
      </section>

      <Separator className="lg:col-span-2" />
    </div>
  )
}
