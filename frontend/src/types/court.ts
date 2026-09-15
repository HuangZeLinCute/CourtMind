/** A 2D point with integer (original video space) or float (display space) coords. */
export interface Point {
  x: number
  y: number
}

/**
 * Court corners in order TL, TR, BR, BL — the order produced by CourtKeyNet
 * and consumed by the Homography/CourtMapper pipeline.
 */
export type Corner = [number, number]
export type Corners = Corner[]

export interface CourtDetectionResponse {
  success: boolean
  detector: string
  corners: Corners | null
  roi_corners?: Corners | null
  mid_height?: number | null
  preview_url?: string | null
  message?: string | null
}

export interface CourtState {
  corners: Corners | null
  detector: string
  roi_corners?: Corners | null
  mid_height?: number | null
  best_frame_path?: string | null
  preview_url?: string | null
}

export const CORNER_LABELS = ["TL", "TR", "BR", "BL"] as const
