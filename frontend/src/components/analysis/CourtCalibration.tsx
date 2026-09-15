import { useMemo, useRef, useState } from "react"

import { CourtOverlay } from "@/components/analysis/CourtOverlay"
import { CourtStatus } from "@/components/analysis/CourtStatus"
import { CornerEditor } from "@/components/analysis/CornerEditor"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { useCourtDetection } from "@/hooks/useCourtDetection"
import { useLanguage } from "@/i18n/LanguageProvider"
import { cornersToDisplay, cornersToOriginal } from "@/lib/coordinates"
import { toFullUrl } from "@/lib/api"
import type { Corners, VideoInfo } from "@/types"

interface CourtCalibrationProps {
  video: VideoInfo
  onConfirmed: (corners: Corners, detectorName: string) => void
}

const DISPLAY_HEIGHT = 480

/**
 * Step 2 — Court calibration.
 *
 * Layout: video frame + SVG overlay (70%) | court status panel (30%).
 * Normal mode: overlay is read-only. Edit mode: drag the four handles.
 * Coordinates stay in display space here and are converted to original video
 * space (backend contract) on Confirm/Apply.
 */
export function CourtCalibration({ video, onConfirmed }: CourtCalibrationProps) {
  const { t } = useLanguage()
  const { court, detecting, error, detect, saveCorners } = useCourtDetection(video.video_id)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState<Corners | null>(null)
  const [saving, setSaving] = useState(false)
  const [legacyOpen, setLegacyOpen] = useState(false)
  const [legacyFile, setLegacyFile] = useState<File | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const originalSize = useMemo(
    () => ({ width: video.width, height: video.height }),
    [video.width, video.height],
  )
  const displaySize = useMemo(() => {
    const ratio = video.height > 0 ? video.width / video.height : 16 / 9
    return { width: Math.round(DISPLAY_HEIGHT * ratio), height: DISPLAY_HEIGHT }
  }, [video.width, video.height])

  // Source of truth for the overlay: edited draft while editing, else detected.
  const sourceCorners = editing ? draft : court?.corners ?? null

  const displayCorners = useMemo(() => {
    if (!sourceCorners || sourceCorners.length !== 4) return null
    return cornersToDisplay(sourceCorners, originalSize, displaySize)
  }, [sourceCorners, originalSize, displaySize])

  const runDetect = async (useLegacy = false, file?: File | null) => {
    setEditing(false)
    const state = await detect(useLegacy, file)
    if (state?.corners) setDraft(null)
  }

  const handleCornerChange = (index: number, x: number, y: number) => {
    if (!editing || !draft) return
    const next = draft.map((c, i) => (i === index ? [Math.round(x), Math.round(y)] as [number, number] : c)) as Corners
    setDraft(next)
  }

  const handleApply = async () => {
    if (!draft) return
    setSaving(true)
    const original = cornersToOriginal(draft, originalSize, displaySize)
    const saved = await saveCorners(original)
    setSaving(false)
    if (saved) {
      setEditing(false)
      setDraft(null)
    }
  }

  const previewUrl = toFullUrl(court?.preview_url)

  return (
    <div className="space-y-4">
      {error && (
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm">
          <span className="text-destructive">{error}</span>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => runDetect(false)}>
              {t("court.retry")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => {
                setLegacyOpen(true)
                setLegacyFile(null)
              }}
            >
              {t("court.legacyDetector")}
            </Button>
          </div>
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,7fr)_minmax(0,3fr)]">
        {/* Preview + overlay */}
        <div className="min-w-0 space-y-3">
          <div
            ref={containerRef}
            className="relative overflow-hidden rounded-lg border bg-black"
            style={{ aspectRatio: `${video.width} / ${video.height}` }}
          >
            {previewUrl ? (
              <>
                <img
                  src={previewUrl}
                  alt="Court detection preview"
                  className="h-full w-full object-contain"
                  draggable={false}
                />
                {displayCorners && (
                  <CourtOverlay
                    corners={displayCorners}
                    displaySize={displaySize}
                    editable={editing}
                    onCornerChange={handleCornerChange}
                  />
                )}
              </>
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
                {detecting ? t("court.detectingText") : t("court.noDetection")}
              </div>
            )}
          </div>

          {editing && draft && (
            <CornerEditor
              original={court?.corners ?? draft}
              current={draft}
              onCancel={() => {
                setDraft(court?.corners ?? null)
                setEditing(false)
              }}
              onReset={() => setDraft(court?.corners ?? null)}
              onApply={handleApply}
              saving={saving}
            />
          )}
        </div>

        {/* Status panel */}
        <div className="min-w-0">
          <CourtStatus
            detecting={detecting}
            courtDetected={!!court?.corners}
            detectorName={court?.detector ?? null}
            cornerCount={court?.corners?.length ?? 0}
            editing={editing}
            onEdit={() => {
              if (court?.corners) {
                setDraft(court.corners)
                setEditing(true)
              }
            }}
            onDetectAgain={() => runDetect(false)}
            onConfirm={() => {
              if (court?.corners) onConfirmed(court.corners, court.detector)
            }}
          />

          {/* Advanced: legacy detector */}
          <div className="mt-4 rounded-lg border bg-card p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t("court.advanced")}</p>
                <p className="text-xs text-muted-foreground">{t("court.advancedDesc")}</p>
              </div>
              <Switch checked={legacyOpen} onCheckedChange={setLegacyOpen} />
            </div>
            {legacyOpen && (
              <div className="mt-3 space-y-3">
                <Separator />
                <div className="space-y-1.5">
                  <Label htmlFor="template">{t("court.templateImage")}</Label>
                  <Input
                    id="template"
                    type="file"
                    accept="image/*"
                    onChange={(e) => setLegacyFile(e.target.files?.[0] ?? null)}
                  />
                  <p className="text-xs text-muted-foreground">{t("court.templateHint")}</p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  className="w-full"
                  disabled={!legacyFile || detecting}
                  onClick={() => runDetect(true, legacyFile)}
                >
                  {t("court.detectLegacy")}
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
