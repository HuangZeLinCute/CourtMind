import { useEffect, useRef, useState } from "react"

import { api } from "@/lib/api"
import type { JobStatusResponse } from "@/types"

const POLL_INTERVAL_MS = 1000

/**
 * Polls GET /api/jobs/{jobId} every second until the job leaves
 * queued/processing. Returns the latest status snapshot.
 */
export function useJobPolling(jobId: string | null, enabled = true) {
  const [status, setStatus] = useState<JobStatusResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const timer = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    if (!jobId || !enabled) return
    setError(null)

    const tick = async () => {
      try {
        const { data } = await api.get<JobStatusResponse>(`/jobs/${jobId}`)
        setStatus(data)
        if (data.status === "completed" || data.status === "failed") {
          if (timer.current) {
            clearInterval(timer.current)
            timer.current = null
          }
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to poll job status")
      }
    }

    void tick()
    timer.current = setInterval(tick, POLL_INTERVAL_MS)
    return () => {
      if (timer.current) {
        clearInterval(timer.current)
        timer.current = null
      }
    }
  }, [jobId, enabled])

  return { status, error }
}
