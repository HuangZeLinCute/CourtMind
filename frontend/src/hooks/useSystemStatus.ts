import { useEffect, useState } from "react"

import { api } from "@/lib/api"
import type { SystemStatus } from "@/types"

export function useSystemStatus() {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    async function load() {
      try {
        const { data } = await api.get<SystemStatus>("/system/status")
        if (!cancelled) {
          setStatus(data)
          setError(null)
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : "Backend unreachable")
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    void load()
    return () => {
      cancelled = true
    }
  }, [])

  return { status, loading, error }
}
