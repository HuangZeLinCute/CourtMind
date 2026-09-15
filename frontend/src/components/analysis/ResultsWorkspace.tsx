import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ResultsOverview } from "@/components/analysis/ResultsOverview"
import { ResultsReport } from "@/components/analysis/ResultsReport"
import { ResultsVideo } from "@/components/analysis/ResultsVideo"
import { useLanguage } from "@/i18n/LanguageProvider"
import type { JobResult } from "@/types"

interface ResultsWorkspaceProps {
  result: JobResult
}

export function ResultsWorkspace({ result }: ResultsWorkspaceProps) {
  const { t } = useLanguage()
  if (result.status !== "completed") {
    return (
      <div className="rounded-lg border bg-card p-6 text-sm text-muted-foreground">
        {t("results.notCompleted")}
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <Tabs defaultValue={result.report ? "report" : "overview"}>
        <TabsList>
          <TabsTrigger value="report">{t("results.tabReport")}</TabsTrigger>
          <TabsTrigger value="overview">{t("results.tabOverview")}</TabsTrigger>
          <TabsTrigger value="video">{t("results.tabVideo")}</TabsTrigger>
        </TabsList>

        <TabsContent value="report">
          <ResultsReport result={result} />
        </TabsContent>

        <TabsContent value="overview">
          <ResultsOverview result={result} />
        </TabsContent>
        <TabsContent value="video">
          <ResultsVideo result={result} />
        </TabsContent>
      </Tabs>
    </div>
  )
}
