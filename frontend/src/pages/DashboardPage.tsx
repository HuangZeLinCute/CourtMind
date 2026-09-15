import { useEffect, useState } from "react"
import { Link } from "react-router-dom"
import { ArrowRight, PlusCircle } from "lucide-react"

import { PageHeader } from "@/components/layout/PageHeader"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { api } from "@/lib/api"
import { formatRelativeTime } from "@/lib/utils"
import { useLanguage } from "@/i18n/LanguageProvider"
import type { HistoryItem } from "@/types"

export function DashboardPage() {
  const { t } = useLanguage()
  const [recent, setRecent] = useState<HistoryItem[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    api
      .get<HistoryItem[]>("/jobs/history")
      .then(({ data }) => {
        if (!cancelled) setRecent(data.slice(0, 6))
      })
      .catch(() => {
        if (!cancelled) setError("Could not load recent analyses")
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="space-y-6">
      <PageHeader
        title={t("dashboard.title")}
        description={t("dashboard.description")}
        actions={
          <Button asChild>
            <Link to="/analysis/new">
              <PlusCircle className="h-4 w-4" /> {t("dashboard.newAnalysis")}
            </Link>
          </Button>
        }
      />

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <section>
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-sm font-semibold">{t("dashboard.recentAnalyses")}</h3>
          <Button variant="ghost" size="sm" asChild>
            <Link to="/history">
              {t("dashboard.viewAll")} <ArrowRight className="ml-1 h-3.5 w-3.5" />
            </Link>
          </Button>
        </div>

        {recent === null ? (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-24" />
            ))}
          </div>
        ) : recent.length === 0 ? (
          <Card>
            <CardContent className="flex flex-col items-center gap-3 p-8 text-center">
              <p className="text-sm text-muted-foreground">
                {t("dashboard.empty")}
              </p>
              <Button asChild size="sm">
                <Link to="/analysis/new">{t("dashboard.startFirst")}</Link>
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {recent.map((item) => (
              <Link
                key={item.job_id}
                to={`/analysis/${item.job_id}`}
                className="rounded-lg border bg-card p-4 transition-colors hover:bg-accent/60"
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="truncate text-sm font-medium" title={item.name}>
                    {item.name}
                  </p>
                  <Badge variant={item.status === "completed" ? "success" : "secondary"}>
                    {item.status}
                  </Badge>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  {formatRelativeTime(item.created_at)}
                </p>
              </Link>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
