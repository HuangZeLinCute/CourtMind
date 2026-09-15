import { useState } from "react"

import { toFullUrl } from "@/lib/api"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { useLanguage } from "@/i18n/LanguageProvider"
import type { JobResult } from "@/types"

/**
 * Heatmaps / scatter grids. Visualizations produced by the pipeline
 * (player heatmaps, trajectories) are listed and open in a lightbox.
 */
export function ResultsHeatmaps({ result }: { result: JobResult }) {
  const { t } = useLanguage()
  const [active, setActive] = useState<string | null>(null)
  const images = result.visualizations ?? []

  if (images.length === 0) {
    return (
      <div className="rounded-lg border bg-card p-6 text-sm text-muted-foreground">
        {t("results.noVisualizations")}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
        {images.map((img, i) => {
          const url = toFullUrl(img)
          if (!url) return null
          return (
            <button
              key={i}
              type="button"
              onClick={() => setActive(url)}
              className="group relative overflow-hidden rounded-lg border bg-black text-left"
            >
              <img
                src={url}
                alt={`Visualization ${i + 1}`}
                className="aspect-video w-full object-contain transition-transform group-hover:scale-[1.02]"
              />
              <span className="absolute bottom-2 right-2 rounded bg-black/60 px-2 py-0.5 text-xs text-white">
                {t("results.clickToEnlarge")}
              </span>
            </button>
          )
        })}
      </div>

      <Dialog open={!!active} onOpenChange={(open) => !open && setActive(null)}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Visualization</DialogTitle>
          </DialogHeader>
          {active && <img src={active} alt="Visualization" className="w-full rounded-md" />}
        </DialogContent>
      </Dialog>
    </div>
  )
}
