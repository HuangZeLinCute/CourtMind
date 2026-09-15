export interface VideoInfo {
  video_id: string
  filename: string
  fps: number
  width: number
  height: number
  duration: number
  file_size_bytes: number
  url?: string | null
}
