import { useEffect, useState } from "react"
import { Link } from "react-router-dom"
import { PlayCircle } from "lucide-react"

import { PageHeader } from "@/components/layout/PageHeader"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { api } from "@/lib/api"
import { formatRelativeTime } from "@/lib/utils"
import { toFullUrl } from "@/lib/api"
import { useLanguage } from "@/i18n/LanguageProvider"
import type { HistoryItem } from "@/types"

export function HistoryPage() {
  const { t } = useLanguage()
  const [items, setItems] = useState<HistoryItem[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .get<HistoryItem[]>("/jobs/history")
      .then(({ data }) => !cancelled && setItems(data))
      .catch(() => !cancelled && setError("Could not load analysis history"))
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="space-y-6">
      <PageHeader title={t("history.title")} description={t("history.description")} />

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      {items === null ? (
        <div className="space-y-2">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-14" />
          ))}
        </div>
      ) : items.length === 0 ? (
        <div className="rounded-lg border bg-card p-8 text-center text-sm text-muted-foreground">
          {t("history.empty")}
        </div>
      ) : (
        <div className="overflow-hidden rounded-lg border bg-card">
          {items.map((item, i) => (
            <div
              key={item.job_id}
              className={`flex items-center gap-4 px-4 py-3 ${i > 0 ? "border-t" : ""}`}
            >
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{item.name}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {item.job_id} · {formatRelativeTime(item.created_at)}
                  {item.duration_sec != null && ` · ${Math.round(item.duration_sec)}s`}
                </p>
              </div>
              <Badge variant={item.status === "completed" ? "success" : "secondary"}>
                {item.status}
              </Badge>
              <Button variant="ghost" size="sm" asChild>
                <Link to={`/analysis/${item.job_id}`}>
                  <PlayCircle className="h-4 w-4" /> {t("history.open")}
                </Link>
              </Button>
              {item.video_url && (
                <Button variant="ghost" size="sm" asChild>
                  <a href={toFullUrl(item.video_url)} target="_blank" rel="noreferrer">
                    {t("history.video")}
                  </a>
                </Button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
