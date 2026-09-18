import { useCallback, useState } from "react"

import { api } from "@/lib/api"
import type { CourtDetectionResponse, CourtState } from "@/types"

/**
 * Court detection state machine for one uploaded video.
 * - loadCourt(): GET /api/videos/{id}/court (already-persisted state, if any).
 * - detect(): POST /api/videos/{id}/detect-court (CourtKeyNet by default,
 *   legacy template detector when useLegacy is set).
 * - saveCorners(): PUT /api/videos/{id}/court (manual corner editor).
 */
export function useCourtDetection(videoId: string | null) {
  const [court, setCourt] = useState<CourtState | null>(null)
  const [detecting, setDetecting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadCourt = useCallback(async () => {
    if (!videoId) return null
    try {
      const { data } = await api.get<CourtState>(`/videos/${videoId}/court`)
      if (data?.corners && data.corners.length === 4) {
        setCourt(data)
        return data
      }
      return null
    } catch {
      // A missing or unreadable court state is not an error: the caller falls
      // back to running detection.
      return null
    }
  }, [videoId])

  const detect = useCallback(
    async (useLegacy = false, templateFile?: File | null) => {
      if (!videoId) return null
      setDetecting(true)
      setError(null)
      try {
        const form = new FormData()
        if (useLegacy && templateFile) form.append("template", templateFile)
        const { data } = await api.post<CourtDetectionResponse>(
          `/videos/${videoId}/detect-court`,
          form,
          {
            params: { use_legacy: String(useLegacy) },
            headers: { "Content-Type": "multipart/form-data" },
          },
        )
        if (!data.success) {
          setError(data.message ?? "Court detection failed")
          return null
        }
        const state: CourtState = {
          corners: data.corners,
          detector: data.detector,
          roi_corners: data.roi_corners,
          mid_height: data.mid_height,
          preview_url: data.preview_url,
        }
        setCourt(state)
        return state
      } catch (e) {
        const msg =
          (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
          (e instanceof Error ? e.message : "Court detection failed")
        setError(msg)
        return null
      } finally {
        setDetecting(false)
      }
    },
    [videoId],
  )

  const saveCorners = useCallback(
    async (corners: [number, number][]) => {
      if (!videoId) return null
      setError(null)
      try {
        const { data } = await api.put<CourtState>(`/videos/${videoId}/court`, {
          corners,
        })
        setCourt(data)
        return data
      } catch (e) {
        const msg =
          (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
          (e instanceof Error ? e.message : "Could not save corners")
        setError(msg)
        return null
      }
    },
    [videoId],
  )

  return { court, detecting, error, loadCourt, detect, saveCorners }
}
