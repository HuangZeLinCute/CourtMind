import { NavLink } from "react-router-dom"
import { BarChart3, Files, History, LayoutDashboard, MessageSquare, PanelLeftClose, PanelLeftOpen, PlusCircle, Settings } from "lucide-react"

import { cn } from "@/lib/utils"
import { useLanguage } from "@/i18n/LanguageProvider"

interface SidebarProps {
  collapsed?: boolean
  onNavigate?: () => void
  onToggleCollapse?: () => void
}

export function Sidebar({ collapsed = false, onNavigate, onToggleCollapse }: SidebarProps) {
  const { t } = useLanguage()

  const NAV_ITEMS = [
    { to: "/", label: t("nav.dashboard"), icon: LayoutDashboard, end: true },
    { to: "/analysis/new", label: t("nav.newAnalysis"), icon: PlusCircle, end: false },
    { to: "/analysis/batch", label: t("nav.batchAnalysis"), icon: Files, end: false },
    { to: "/chat", label: t("nav.chat"), icon: MessageSquare, end: false },
    { to: "/history", label: t("nav.history"), icon: History, end: false },
    { to: "/settings", label: t("nav.settings"), icon: Settings, end: false },
  ]

  return (
      <aside
        className={cn(
          "flex h-full flex-col border-r bg-muted/[0.18] transition-[width] duration-300",
          collapsed ? "w-[64px]" : "w-[240px]",
        )}
      >
        <div className={cn("flex h-16 items-center gap-2.5 border-b px-3", collapsed && "justify-center px-0")}>
          {collapsed && onToggleCollapse ? (
            <button type="button" onClick={onToggleCollapse} className="mx-auto flex h-10 w-10 items-center justify-center rounded-xl text-muted-foreground transition-colors hover:bg-accent hover:text-foreground" aria-label={t("nav.openSidebar")} title={t("nav.openSidebar")}><PanelLeftOpen className="h-4 w-4" /></button>
          ) : <>
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm">
              <BarChart3 className="h-4 w-4 text-primary-foreground" />
            </div>
            <span className="min-w-0 flex-1 overflow-hidden whitespace-nowrap text-sm font-semibold tracking-tight">CourtMind</span>
            {onToggleCollapse && <button type="button" onClick={onToggleCollapse} className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl text-muted-foreground transition-colors hover:bg-accent hover:text-foreground" aria-label={t("nav.closeSidebar")} title={t("nav.closeSidebar")}><PanelLeftClose className="h-4 w-4" /></button>}
          </>}
        </div>

        <nav className="flex-1 space-y-1.5 overflow-y-auto p-3">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon
            const link = (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                onClick={onNavigate}
                aria-label={item.label}
                title={collapsed ? item.label : undefined}
                className={({ isActive }) =>
                  cn(
                    "flex h-10 items-center gap-3 rounded-xl px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-accent/80 hover:text-foreground",
                    isActive && "bg-primary/10 text-primary shadow-sm ring-1 ring-primary/10",
                    collapsed && "mx-auto w-10 justify-center p-0",
                  )
                }
              >
                <Icon className="h-4 w-4 shrink-0" />
                <span className={cn("overflow-hidden whitespace-nowrap transition-[max-width,opacity,transform] duration-300", collapsed ? "max-w-0 -translate-x-2 opacity-0" : "max-w-[150px] translate-x-0 opacity-100")}>{item.label}</span>
              </NavLink>
            )
            return link
          })}
        </nav>

        <div className="border-t p-3">
          <div className="relative min-h-8 overflow-hidden">
            <p className={cn("absolute inset-0 whitespace-nowrap text-xs leading-relaxed text-muted-foreground transition-[opacity,transform] duration-300", collapsed ? "-translate-x-2 opacity-0" : "translate-x-0 opacity-100")}>
              AI badminton match analysis
              <br />
              v1.0.0
            </p>
            <span className={cn("absolute inset-0 block pt-2 text-center text-xs text-muted-foreground transition-opacity duration-300", collapsed ? "opacity-100" : "opacity-0")}>v1</span>
          </div>
        </div>
      </aside>
  )
}
