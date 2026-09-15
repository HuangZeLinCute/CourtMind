import { useCallback, useState } from "react"

import { api } from "@/lib/api"
import type {
  AnalysisOptions,
  AnalysisStartResponse,
  JobResult,
  JobStatusResponse,
} from "@/types"

/**
 * Analysis job lifecycle: start a job, poll it, fetch results.
 * The polling itself lives in useJobPolling; this hook wires the pieces the
 * New Analysis page needs.
 */
export function useAnalysis(videoId: string | null) {
  const [jobId, setJobId] = useState<string | null>(null)
  const [status, setStatus] = useState<JobStatusResponse | null>(null)
  const [result, setResult] = useState<JobResult | null>(null)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const start = useCallback(
    async (options: AnalysisOptions, corners?: [number, number][], name?: string) => {
      if (!videoId) return null
      setStarting(true)
      setError(null)
      try {
        const { data } = await api.post<AnalysisStartResponse>(
          `/videos/${videoId}/analyze`,
          { corners: corners ?? null, options, name: name?.trim() || null },
        )
        setJobId(data.job_id)
        return data.job_id
      } catch (e) {
        const msg =
          (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
          (e instanceof Error ? e.message : "Could not start analysis")
        setError(msg)
        return null
      } finally {
        setStarting(false)
      }
    },
    [videoId],
  )

  const refreshStatus = useCallback(async (id: string) => {
    try {
      const { data } = await api.get<JobStatusResponse>(`/jobs/${id}`)
      setStatus(data)
      return data
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load job status")
      return null
    }
  }, [])

  const loadResult = useCallback(async (id: string) => {
    try {
      const { data } = await api.get<JobResult>(`/jobs/${id}/results`)
      setResult(data)
      return data
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load results")
      return null
    }
  }, [])

  return { jobId, status, result, starting, error, start, refreshStatus, loadResult }
}
