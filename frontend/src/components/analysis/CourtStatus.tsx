import { CheckCircle2, Loader2, XCircle } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { useLanguage } from "@/i18n/LanguageProvider"

interface CourtStatusProps {
  detecting: boolean
  courtDetected: boolean
  detectorName: string | null
  cornerCount: number
  editing: boolean
  onEdit: () => void
  onDetectAgain: () => void
  onConfirm: () => void
  disabled?: boolean
}

export function CourtStatus({
  detecting,
  courtDetected,
  detectorName,
  cornerCount,
  editing,
  onEdit,
  onDetectAgain,
  onConfirm,
  disabled,
}: CourtStatusProps) {
  const { t } = useLanguage()
  return (
    <div className="flex flex-col gap-4 rounded-lg border bg-card p-5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">{t("court.status")}</h3>
        {detecting ? (
          <Badge variant="secondary">
            <Loader2 className="mr-1 h-3 w-3 animate-spin" /> {t("court.detecting")}
          </Badge>
        ) : courtDetected ? (
          <Badge variant="success">
            <CheckCircle2 className="mr-1 h-3 w-3" /> {t("court.detected")}
          </Badge>
        ) : (
          <Badge variant="destructive">
            <XCircle className="mr-1 h-3 w-3" /> {t("court.missing")}
          </Badge>
        )}
      </div>

      <Separator />

      <div className="space-y-2 text-sm">
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">{t("court.detector")}</span>
          <span className="font-mono text-xs">{detectorName ?? "—"}</span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">{t("court.corners")}</span>
          <span className={courtDetected ? "text-emerald-500" : ""}>
            {cornerCount}/4
          </span>
        </div>
        {editing && (
          <p className="rounded-md bg-amber-500/10 px-2 py-1.5 text-xs text-amber-600 dark:text-amber-400">
            {t("court.editHint")}
          </p>
        )}
      </div>

      <div className="mt-auto space-y-2">
        <Button className="w-full" onClick={onConfirm} disabled={!courtDetected || disabled}>
          {t("court.confirm")}
        </Button>
        <div className="grid grid-cols-2 gap-2">
          <Button variant="outline" onClick={onEdit} disabled={!courtDetected}>
            {editing ? t("court.exitEdit") : t("court.editCorners")}
          </Button>
          <Button variant="outline" onClick={onDetectAgain} disabled={detecting}>
            {t("court.detectAgain")}
          </Button>
        </div>
      </div>
    </div>
  )
}

export function CourtStatusSkeleton() {
  return (
    <div className="flex flex-col gap-4 rounded-lg border bg-card p-5">
      <Skeleton className="h-5 w-32" />
      <Skeleton className="h-px w-full" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-4 w-full" />
      <Skeleton className="h-10 w-full" />
    </div>
  )
}
