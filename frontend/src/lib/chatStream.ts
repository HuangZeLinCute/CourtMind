import { API_BASE } from "@/lib/api"
import type { ChatMessage, ChatProvider } from "@/types"

type StreamEvent =
  | { type: "delta"; content: string }
  | { type: "done"; message: ChatMessage }
  | { type: "error"; message: string }

export async function streamReply(
  jobId: string, provider: ChatProvider, message: string,
  signal: AbortSignal, onDelta: (text: string) => void,
): Promise<ChatMessage> {
  const response = await fetch(`${API_BASE}/chat/${jobId}/stream`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ provider, message }), signal,
  })
  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  if (!response.body) throw new Error("Streaming is unavailable")
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  try {
    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      let end: number
      while ((end = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, end).trim()
        buffer = buffer.slice(end + 1)
        if (!line.startsWith("data:")) continue
        const event = JSON.parse(line.slice(5).trim()) as StreamEvent
        if (event.type === "delta") onDelta(event.content)
        if (event.type === "error") throw new Error(event.message)
        if (event.type === "done") return event.message
      }
      if (done) throw new Error("回复连接已中断，请重试。 / Connection interrupted.")
    }
  } finally {
    await reader.cancel().catch(() => undefined)
    reader.releaseLock()
  }
}
