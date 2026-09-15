import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react"

export type ThemeMode = "light" | "dark" | "system"

interface ThemeContextValue {
  theme: ThemeMode
  setTheme: (theme: ThemeMode) => void
  /** The actually-applied theme (system resolves to light/dark). */
  resolved: "light" | "dark"
}

const ThemeContext = createContext<ThemeContextValue>({
  theme: "light",
  setTheme: () => undefined,
  resolved: "light",
})

function systemPrefersDark(): boolean {
  return typeof window !== "undefined"
    && window.matchMedia("(prefers-color-scheme: dark)").matches
}

function applyClass(mode: ThemeMode) {
  const dark = mode === "dark" || (mode === "system" && systemPrefersDark())
  document.documentElement.classList.toggle("dark", dark)
}

/**
 * Global theme provider. Default is LIGHT. "system" follows the OS preference
 * and stays in sync while the OS preference changes.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<ThemeMode>("light")

  const setThemeWithTransition = (next: ThemeMode) => {
    const root = document.documentElement
    root.classList.add("theme-transition")
    setTheme(next)
    window.setTimeout(() => root.classList.remove("theme-transition"), 420)
  }

  useEffect(() => {
    applyClass(theme)
    if (theme !== "system") return

    const media = window.matchMedia("(prefers-color-scheme: dark)")
    const onChange = () => applyClass("system")
    media.addEventListener("change", onChange)
    return () => media.removeEventListener("change", onChange)
  }, [theme])

  const resolved: "light" | "dark" =
    theme === "system" ? (systemPrefersDark() ? "dark" : "light") : theme

  return (
    <ThemeContext.Provider value={{ theme, setTheme: setThemeWithTransition, resolved }}>
      {children}
    </ThemeContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useTheme() {
  return useContext(ThemeContext)
}
