import { useEffect, useState } from "react"

import { useTheme } from "@/components/theme/ThemeProvider"
import { useLanguage } from "@/i18n/LanguageProvider"
import { PageHeader } from "@/components/layout/PageHeader"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import type { ThemeMode } from "@/components/theme/ThemeProvider"
import { ModelsPage } from "@/pages/ModelsPage"
import { api } from "@/lib/api"
import type { ChatProvider, ChatProviderInfo } from "@/types"

export function SettingsPage() {
  const { theme, setTheme } = useTheme()
  const { language, setLanguage, t } = useLanguage()
  const [chatProviders, setChatProviders] = useState<ChatProviderInfo[]>([])
  const [chatProvider, setChatProvider] = useState<ChatProvider>(() => (localStorage.getItem("cm-chat-provider") as ChatProvider | null) ?? "agnes")

  useEffect(() => {
    const controller = new AbortController()
    void api.get<{ providers: ChatProviderInfo[] }>("/chat/config", { signal: controller.signal }).then(({ data }) => {
      if (controller.signal.aborted) return
      setChatProviders(data.providers)
      const selected = data.providers.find((item) => item.id === chatProvider && item.configured)
        ?? data.providers.find((item) => item.configured)
      if (selected) {
        setChatProvider(selected.id)
        localStorage.setItem("cm-chat-provider", selected.id)
      }
    }).catch(() => undefined)
    return () => controller.abort()
  }, [])

  return (
    <div className="mx-auto max-w-5xl space-y-6">
      <PageHeader title={t("settings.title")} description={t("settings.description")} />

      <div className="space-y-6 rounded-lg border bg-card p-6">
        <div className="space-y-2">
          <Label>{t("settings.theme")}</Label>
          <Select value={theme} onValueChange={(v) => setTheme(v as ThemeMode)}>
            <SelectTrigger className="w-full sm:w-56">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="light">{t("topbar.light")}</SelectItem>
              <SelectItem value="dark">{t("topbar.dark")}</SelectItem>
              <SelectItem value="system">{t("topbar.system")}</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">{t("settings.themeDesc")}</p>
        </div>

        <Separator />

        <div className="space-y-2">
          <Label>{t("settings.language")}</Label>
          <Select value={language} onValueChange={(v) => setLanguage(v as "zh" | "en")}>
            <SelectTrigger className="w-full sm:w-56">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="zh">简体中文</SelectItem>
              <SelectItem value="en">English</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">{t("settings.languageDesc")}</p>
        </div>

        <Separator />

        <div className="space-y-2">
          <Label>{language === "zh" ? "对局分析模型" : "Match analysis model"}</Label>
          <Select value={chatProvider} onValueChange={(value) => { const next = value as ChatProvider; setChatProvider(next); localStorage.setItem("cm-chat-provider", next) }}>
            <SelectTrigger className="w-full sm:w-72">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {chatProviders.map((item) => <SelectItem key={item.id} value={item.id} disabled={!item.configured}>{item.label} · {item.model}{!item.configured && (language === "zh" ? "（未配置）" : " (not configured)")}</SelectItem>)}
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">{language === "zh" ? "对局分析聊天会使用这里选择的 AI 模型。" : "Match analysis chat uses the AI model selected here."}</p>
        </div>

        <Separator />

        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">{t("settings.advanced")}</p>
            <p className="text-xs text-muted-foreground">{t("settings.advancedDesc")}</p>
          </div>
          <Button variant="outline" size="sm" disabled>
            {t("settings.openAdvanced")}
          </Button>
        </div>
      </div>

      <section className="space-y-3">
        <div>
          <h3 className="text-base font-semibold">{t("models.title")}</h3>
          <p className="mt-1 text-sm text-muted-foreground">{t("models.description")}</p>
        </div>
        <ModelsPage embedded />
      </section>
    </div>
  )
}
