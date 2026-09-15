import { useState } from "react"
import { Outlet, useLocation } from "react-router-dom"

import { MobileSidebar } from "@/components/layout/MobileSidebar"
import { Sidebar } from "@/components/layout/Sidebar"
import { Topbar } from "@/components/layout/Topbar"
import { cn } from "@/lib/utils"

export function AppShell() {
  const location = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)
  const [sidebarClosed, setSidebarClosed] = useState(() => localStorage.getItem("gb-sidebar-closed") === "true")
  const toggleSidebar = () => setSidebarClosed((closed) => {
    localStorage.setItem("gb-sidebar-closed", String(!closed))
    return !closed
  })
  const isChat = location.pathname === "/chat"

  return (
    <div className="flex h-dvh overflow-hidden bg-background">
      <div
        className={cn(
          "hidden shrink-0 overflow-hidden transition-[width] duration-300 ease-in-out lg:block",
          sidebarClosed ? "w-[64px]" : "w-[240px]",
        )}
      >
        <div className={cn("h-full transition-[width] duration-300 ease-in-out", sidebarClosed ? "w-[64px]" : "w-[240px]")}>
          <Sidebar collapsed={sidebarClosed} onToggleCollapse={toggleSidebar} />
        </div>
      </div>

      <MobileSidebar open={mobileOpen} onOpenChange={setMobileOpen} />

      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <Topbar floating onOpenSidebar={() => setMobileOpen(true)} />
        <main className={cn("min-h-0 flex-1", isChat ? "overflow-hidden" : "overflow-y-auto")}>
          <div className={cn("mx-auto w-full", isChat ? "h-full min-h-0" : "max-w-[1500px] px-4 pb-6 pt-16 sm:px-6 lg:px-8 lg:py-6")}>
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}
