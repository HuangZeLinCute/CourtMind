import { createBrowserRouter, Navigate } from "react-router-dom"

import { AppShell } from "@/components/layout/AppShell"
import { AnalysisResultPage } from "@/pages/AnalysisResultPage"
import { BatchAnalysisPage } from "@/pages/BatchAnalysisPage"
import { ChatPage } from "@/pages/ChatPage"
import { DashboardPage } from "@/pages/DashboardPage"
import { HistoryPage } from "@/pages/HistoryPage"
import { NewAnalysisPage } from "@/pages/NewAnalysisPage"
import { SettingsPage } from "@/pages/SettingsPage"

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <DashboardPage /> },
      { path: "analysis/new", element: <NewAnalysisPage /> },
      { path: "analysis/batch", element: <BatchAnalysisPage /> },
      { path: "analysis/:jobId", element: <AnalysisResultPage /> },
      { path: "chat", element: <ChatPage /> },
      { path: "history", element: <HistoryPage /> },
      { path: "models", element: <Navigate to="/settings" replace /> },
      { path: "settings", element: <SettingsPage /> },
      { path: "*", element: <Navigate to="/" replace /> },
    ],
  },
])
