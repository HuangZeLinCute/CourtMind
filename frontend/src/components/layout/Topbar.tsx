import { PanelLeft } from "lucide-react"

import { Button } from "@/components/ui/button"
import { useLanguage } from "@/i18n/LanguageProvider"
import { cn } from "@/lib/utils"

interface TopbarProps {
  title?: string
  floating?: boolean
  onOpenSidebar: () => void
}

export function Topbar({ title, floating = false, onOpenSidebar }: TopbarProps) {
  const { t } = useLanguage()

  return (
    <header className={cn("flex shrink-0 items-center justify-between", floating ? "absolute left-3 top-2 z-30 h-10 lg:hidden" : "h-14 border-b px-4")}>
      <div className={cn("flex items-center gap-3", floating && "contents")}>
        <Button variant="ghost" size="icon" className="lg:hidden" onClick={onOpenSidebar} aria-label={t("nav.openSidebar")}>
          <PanelLeft className="h-4 w-4" />
        </Button>
        {title && <h1 className="text-sm font-semibold">{title}</h1>}
      </div>
    </header>
  )
}
