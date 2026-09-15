import axios from "axios"

/**
 * Single Axios instance for the whole frontend.
 * The backend CORS allow-list includes http://localhost:5173 / 127.0.0.1:5173.
 * Override with VITE_API_BASE when the backend is deployed elsewhere.
 */
export const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8000/api"

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 120_000,
})

export function toFullUrl(path: string | null | undefined): string | undefined {
  if (!path) return undefined
  if (path.startsWith("http")) return path
  const base = (import.meta.env.VITE_API_BASE as string | undefined) ?? "http://127.0.0.1:8000"
  return `${base}${path}`
}
