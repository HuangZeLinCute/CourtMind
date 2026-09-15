import { useCallback, useRef } from "react"

import { cn } from "@/lib/utils"
import type { Corners } from "@/types/court"

export interface Size {
  width: number
  height: number
}

interface CourtOverlayProps {
  /** Corners in DISPLAY coordinates (already scaled by lib/coordinates). */
  corners: Corners
  /** Display space size — the SVG viewBox matches it, so corners map 1:1. */
  displaySize: Size
  /** Whether the user may drag the handles. */
  editable: boolean
  onCornerChange?: (index: number, x: number, y: number) => void
  /** Show the TL/TR/BR/BL labels on handles. */
  showLabels?: boolean
  className?: string
}

/**
 * SVG overlay layered absolutely on top of the video frame.
 *
 * The viewBox equals the display size (e.g. 853x480), so corners coming from
 * lib/coordinates (display space) map 1:1 — no 0..1000 rescaling. Coordinates
 * are converted back to original video space by the caller before the API.
 */
export function CourtOverlay({
  corners,
  displaySize,
  editable = false,
  onCornerChange,
  showLabels = true,
  className,
}: CourtOverlayProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const dragIndex = useRef<number | null>(null)

  const toSvgPoint = useCallback(
    (clientX: number, clientY: number) => {
      const svg = svgRef.current
      if (!svg) return null
      const rect = svg.getBoundingClientRect()
      if (rect.width <= 0 || rect.height <= 0) return null
      const x = ((clientX - rect.left) / rect.width) * displaySize.width
      const y = ((clientY - rect.top) / rect.height) * displaySize.height
      return { x, y }
    },
    [displaySize],
  )

  const handlePointerDown = useCallback(
    (e: React.PointerEvent, index: number) => {
      if (!editable) return
      e.preventDefault()
      e.stopPropagation()
      dragIndex.current = index
      ;(e.target as Element).setPointerCapture?.(e.pointerId)
    },
    [editable],
  )

  const handlePointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (dragIndex.current === null || !onCornerChange) return
      const p = toSvgPoint(e.clientX, e.clientY)
      if (p) onCornerChange(dragIndex.current, p.x, p.y)
    },
    [onCornerChange, toSvgPoint],
  )

  const endDrag = useCallback(() => {
    dragIndex.current = null
  }, [])

  if (corners.length !== 4) return null

  const [tl, tr, br, bl] = corners
  const polygonPoints = `${tl[0]},${tl[1]} ${tr[0]},${tr[1]} ${br[0]},${br[1]} ${bl[0]},${bl[1]}`
  const labels = ["TL", "TR", "BR", "BL"]

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${displaySize.width} ${displaySize.height}`}
      preserveAspectRatio="none"
      className={cn("absolute inset-0 h-full w-full overflow-visible", className)}
      onPointerMove={handlePointerMove}
      onPointerUp={endDrag}
      onPointerLeave={endDrag}
    >
      <polygon
        points={polygonPoints}
        fill="rgba(16,185,129,0.12)"
        stroke={editable ? "#10b981" : "rgba(16,185,129,0.85)"}
        strokeWidth={editable ? 4 : 3}
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
      />
      {[tl, tr, br, bl].map(([x, y], i) => (
        <g key={labels[i]}>
          <circle
            cx={x}
            cy={y}
            r={editable ? 26 : 16}
            fill="rgba(239,68,68,0.9)"
            stroke="#ffffff"
            strokeWidth={2}
            vectorEffect="non-scaling-stroke"
            style={{ cursor: editable ? "grab" : "default" }}
            onPointerDown={(e) => handlePointerDown(e, i)}
          />
          {showLabels && (
            <text
              x={x}
              y={y - 22}
              textAnchor="middle"
              fill="#ffffff"
              fontSize="22"
              fontWeight={700}
              style={{ pointerEvents: "none", userSelect: "none" }}
            >
              {labels[i]}
            </text>
          )}
        </g>
      ))}
    </svg>
  )
}
