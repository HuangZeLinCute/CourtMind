import { RotateCcw, X } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { useLanguage } from "@/i18n/LanguageProvider"

interface CornerEditorProps {
  original: [number, number][]
  current: [number, number][]
  onCancel: () => void
  onReset: () => void
  onApply: () => void
  saving?: boolean
}

/**
 * Toolbar shown while in edit mode. The actual dragging happens on the
 * CourtOverlay handles; this bar only manages cancel/reset/apply.
 */
export function CornerEditor({
  original,
  current,
  onCancel,
  onReset,
  onApply,
  saving,
}: CornerEditorProps) {
  const { t } = useLanguage()
  const dirty = JSON.stringify(original) !== JSON.stringify(current)
  return (
    <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-amber-500/40 bg-amber-500/5 px-4 py-2.5">
      <div className="text-xs text-muted-foreground">{t("court.editMode")}</div>
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel} disabled={saving}>
          <X className="h-3.5 w-3.5" /> {t("court.cancel")}
        </Button>
        <Button variant="outline" size="sm" onClick={onReset} disabled={saving}>
          <RotateCcw className="h-3.5 w-3.5" /> {t("court.reset")}
        </Button>
        <Separator orientation="vertical" className="h-5" />
        <Button size="sm" onClick={onApply} disabled={!dirty || saving}>
          {saving ? t("court.saving") : t("court.apply")}
        </Button>
      </div>
    </div>
  )
}
