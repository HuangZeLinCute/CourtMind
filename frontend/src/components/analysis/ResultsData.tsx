import { useState } from "react"
import { Download } from "lucide-react"

import { toFullUrl } from "@/lib/api"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Separator } from "@/components/ui/separator"
import { useLanguage } from "@/i18n/LanguageProvider"
import type { JobResult } from "@/types"

/**
 * Data tab — raw artifacts (detections.jsonl / metadata.json) with a
 * collapsible preview. No giant JSON dump by default.
 */
export function ResultsData({ result }: { result: JobResult }) {
  const { t } = useLanguage()
  const [preview, setPreview] = useState<{ title: string; content: string } | null>(null)

  const detectionsUrl = toFullUrl(result.detections_url)
  const metadataUrl = toFullUrl(result.metadata_url)
  const reportUrl = toFullUrl(result.report_url)

  const loadPreview = async (url: string | undefined, title: string) => {
    if (!url) return
    try {
      const resp = await fetch(url)
      const text = await resp.text()
      setPreview({ title, content: text.slice(0, 8000) })
    } catch {
      setPreview({ title, content: "Could not load preview." })
    }
  }

  const rows = [
    { label: "detections.jsonl", url: detectionsUrl, title: "detections.jsonl" },
    { label: "metadata.json", url: metadataUrl, title: "metadata.json" },
    { label: "match_report.json", url: reportUrl, title: "match_report.json" },
  ]

  return (
    <div className="space-y-3">
      <div className="grid gap-3 sm:grid-cols-2">
        {rows.map((row) => (
          <div key={row.label} className="flex items-center justify-between rounded-lg border bg-card p-4">
            <span className="font-mono text-sm">{row.label}</span>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={!row.url}
                onClick={() => void loadPreview(row.url, row.title)}
              >
                {t("results.preview")}
              </Button>
              {row.url ? (
                <Button variant="outline" size="sm" asChild>
                  <a href={row.url} download>
                    <Download className="h-3.5 w-3.5" />
                  </a>
                </Button>
              ) : (
                <Button variant="outline" size="sm" disabled>
                  <Download className="h-3.5 w-3.5" />
                </Button>
              )}
            </div>
          </div>
        ))}
      </div>

      <p className="text-xs text-muted-foreground">{t("results.dataHint")}</p>

      <Separator />

      <Dialog open={!!preview} onOpenChange={(open) => !open && setPreview(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="font-mono text-sm">{preview?.title}</DialogTitle>
          </DialogHeader>
          <pre className="max-h-[60vh] overflow-auto rounded-md bg-muted/60 p-4 text-xs leading-relaxed">
            {preview?.content}
          </pre>
        </DialogContent>
      </Dialog>
    </div>
  )
}
