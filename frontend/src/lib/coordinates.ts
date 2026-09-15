/**
 * Coordinate conversion between ORIGINAL video pixel space (what the backend
 * always speaks) and the display space of the court overlay.
 *
 * Rules:
 *  - Backend always returns / accepts ORIGINAL video coordinates.
 *  - The SVG overlay renders in display space; every rendered point is derived
 *    from an original point via toDisplay, and every drag result is converted
 *    back via toOriginal before hitting the API.
 */
import type { Corner, Point } from "@/types/court"

export interface Size {
  width: number
  height: number
}

/** Original video -> display space (e.g. 1920x1080 -> 960x540). */
export function toDisplay(point: Point, original: Size, display: Size): Point {
  if (original.width <= 0 || original.height <= 0) return { x: 0, y: 0 }
  return {
    x: (point.x * display.width) / original.width,
    y: (point.y * display.height) / original.height,
  }
}

/** Display space -> original video space. */
export function toOriginal(point: Point, original: Size, display: Size): Point {
  if (display.width <= 0 || display.height <= 0) return { x: 0, y: 0 }
  return {
    x: (point.x * original.width) / display.width,
    y: (point.y * original.height) / display.height,
  }
}

/** Convert an array of original corners to display coordinates. */
export function cornersToDisplay(
  corners: Corner[],
  original: Size,
  display: Size,
): Corner[] {
  return corners.map(([x, y]) => {
    const p = toDisplay({ x, y }, original, display)
    return [p.x, p.y]
  })
}

/** Convert an array of display corners back to original coordinates. */
export function cornersToOriginal(
  corners: Corner[],
  original: Size,
  display: Size,
): Corner[] {
  return corners.map(([x, y]) => {
    const p = toOriginal({ x, y }, original, display)
    return [Math.round(p.x), Math.round(p.y)]
  })
}
