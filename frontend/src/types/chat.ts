export type ChatProvider = "agnes" | "deepseek"

export interface ChatProviderInfo {
  id: ChatProvider
  label: string
  model: string
  configured: boolean
}

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
  created_at: number
  frames?: ChatFrame[]
}

export interface ChatFrame {
  image_url: string
  frame: number
  time_sec: number
  caption: string
  player?: string | null
  confidence?: number | null
}

export interface ChatMatch {
  job_id: string
  name: string
  created_at?: number | null
  video_url?: string | null
  summary?: {
    rally_count?: number
    total_hits?: number
    longest_rally_hits?: number
  } | null
}

export interface ChatHistory {
  job_id: string
  provider: ChatProvider
  messages: ChatMessage[]
}

export interface ChatResponse {
  job_id: string
  provider: ChatProvider
  model: string
  message: ChatMessage
}
