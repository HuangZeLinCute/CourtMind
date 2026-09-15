import { RouterProvider } from "react-router-dom"
import { Toaster } from "sonner"

import { ThemeProvider } from "@/components/theme/ThemeProvider"
import { LanguageProvider } from "@/i18n/LanguageProvider"
import { router } from "@/router"

export default function App() {
  return (
    <ThemeProvider>
      <LanguageProvider>
        <RouterProvider router={router} />
        <Toaster position="top-right" richColors />
      </LanguageProvider>
    </ThemeProvider>
  )
}
