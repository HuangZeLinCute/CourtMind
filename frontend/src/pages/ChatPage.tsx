import { useEffect, useRef, useState } from "react"
import { ArrowDown, ArrowUp, ArrowUpRight, Camera, ChartNoAxesCombined, Check, Copy, Loader2, MessageSquare, RotateCcw, Sparkles, Square } from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { api, toFullUrl } from "@/lib/api"
import { streamReply } from "@/lib/chatStream"
import { cn } from "@/lib/utils"
import { useLanguage } from "@/i18n/LanguageProvider"
import type { ChatHistory, ChatMatch, ChatMessage, ChatProvider, ChatProviderInfo } from "@/types"

type DisplayMessage = ChatMessage & { incomplete?: boolean }

function cleanInternalReferences(content: string) {
  return content
    .replace(/`?\[(?:report|metadata|frame_state)(?:\.[A-Za-z0-9_]+|\[\d+\])+\]`?/g, "")
    .replace(/[ \t]+([，。；：,.!?])/g, "$1")
}

function errorText(error: unknown) {
  return (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail
    ?? (error instanceof Error ? error.message : "Request failed")
}

// Minimal, safe Markdown renderer for model answers. Supports headings,
// bullet/ordered lists, horizontal rules, **bold**, [report.*] tags and GFM
// tables (| a | b | + |---|---|). Raw HTML from the model is never rendered.
type Align = "left" | "center" | "right"
type Block =
  | { kind: "heading" | "paragraph" | "bullet" | "ordered"; text: string; marker?: string }
  | { kind: "hr" }
  | { kind: "table"; headers: string[]; align: Align[]; rows: string[][] }

const TABLE_SEPARATOR = /^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$/

function splitTableRow(line: string): string[] {
  let s = line.trim()
  if (s.startsWith("|")) s = s.slice(1)
  if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1)
  return s.split(/(?<!\\)\|/).map((cell) => cell.trim().replace(/\\\|/g, "|"))
}

function parseBlocks(content: string): Block[] {
  const lines = content.split("\n")
  const blocks: Block[] = []
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (!line.trim()) { i += 1; continue }
    if (/^\s*([-*_])\1{2,}\s*$/.test(line)) { blocks.push({ kind: "hr" }); i += 1; continue }
    // GFM table: header row (contains |) followed by a separator row.
    if (line.includes("|") && i + 1 < lines.length && TABLE_SEPARATOR.test(lines[i + 1])) {
      const headers = splitTableRow(line)
      const sepAlign = splitTableRow(lines[i + 1]).map((cell): Align => {
        const left = cell.startsWith(":")
        const right = cell.endsWith(":")
        return left && right ? "center" : right ? "right" : "left"
      })
      const align: Align[] = headers.map((_, k) => sepAlign[k] ?? "left")
      const rows: string[][] = []
      let j = i + 2
      while (j < lines.length && lines[j].trim() && lines[j].includes("|")) {
        const cells = splitTableRow(lines[j])
        while (cells.length < headers.length) cells.push("")
        rows.push(cells.slice(0, headers.length))
        j += 1
      }
      blocks.push({ kind: "table", headers, align, rows })
      i = j
      continue
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(line)
    if (heading) { blocks.push({ kind: "heading", text: heading[2].trim() }); i += 1; continue }
    const bullet = /^[-*]\s+(.*)$/.exec(line)
    if (bullet) { blocks.push({ kind: "bullet", text: bullet[1].trim() }); i += 1; continue }
    const ordered = /^(\d+)[.)]\s+(.*)$/.exec(line)
    if (ordered) { blocks.push({ kind: "ordered", text: ordered[2].trim(), marker: ordered[1] }); i += 1; continue }
    blocks.push({ kind: "paragraph", text: line.trim() })
    i += 1
  }
  return blocks
}

function Inline({ text }: { text: string }) {
  return <>{text.split(/(\*\*[^*]+\*\*)/g).map((part, i) => {
    if (!part) return null
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={i} className="font-semibold">{part.slice(2, -2)}</strong>
    return part
  })}</>
}

// Format common report text safely, without accepting HTML from the model.
function Answer({ content }: { content: string }) {
  const visible = cleanInternalReferences(content).replace(/\[\[FRAME:[\s\S]*?\]\]/g, "").replace(/\[\[FRAME:[^\]]*$/g, "")
  const blocks = parseBlocks(visible)
  return (
    <div className="space-y-2.5 text-[14px] leading-7 sm:text-[15px]">
      {blocks.map((block, index) => {
        if (block.kind === "hr") return <hr key={index} className="my-3 border-border/70" />
        if (block.kind === "table") {
          return (
            <div key={index} className="my-2 overflow-x-auto rounded-lg border">
              <table className="w-full border-collapse text-[13px]">
                <thead>
                  <tr className="bg-muted/60">
                    {block.headers.map((head, k) => (
                      <th key={k} style={{ textAlign: block.align[k] }} className="whitespace-nowrap border-b px-3 py-2 font-semibold">
                        <Inline text={head} />
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {block.rows.map((row, r) => (
                    <tr key={r} className="even:bg-muted/25">
                      {row.map((cell, c) => (
                        <td key={c} style={{ textAlign: block.align[c] }} className="border-b border-border/50 px-3 py-1.5 align-top [overflow-wrap:anywhere] last:border-b-0">
                          <Inline text={cell} />
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        }
        if (block.kind === "heading") {
          return <p key={index} className="pt-2 font-semibold"><Inline text={block.text} /></p>
        }
        if (block.kind === "bullet") {
          return (
            <p key={index} className="flex gap-2 break-words [overflow-wrap:anywhere]">
              <span className="select-none text-primary">•</span><span><Inline text={block.text} /></span>
            </p>
          )
        }
        if (block.kind === "ordered") {
          return (
            <p key={index} className="flex gap-2 break-words [overflow-wrap:anywhere]">
              <span className="select-none font-medium text-primary">{block.marker}.</span><span><Inline text={block.text} /></span>
            </p>
          )
        }
        return <p key={index} className="break-words [overflow-wrap:anywhere]"><Inline text={block.text} /></p>
      })}
    </div>
  )
}

function FrameClip({ videoUrl, imageUrl, timeSec, caption, frame, zh }: { videoUrl?: string; imageUrl: string; timeSec: number; caption: string; frame: number; zh: boolean }) {
  const clipRef = useRef<HTMLVideoElement>(null)
  const start = Math.max(0, timeSec - 0.6)
  const end = timeSec + 4
  useEffect(() => {
    const video = clipRef.current
    if (video?.readyState) video.currentTime = start
  }, [start, videoUrl])
  return <figure className="mt-4 max-w-xl overflow-hidden rounded-2xl border bg-muted/20 shadow-sm">
    {videoUrl ? <video ref={clipRef} controls preload="metadata" poster={imageUrl} src={`${videoUrl}#t=${start.toFixed(3)},${end.toFixed(3)}`} onLoadedMetadata={(event) => { event.currentTarget.currentTime = start }} onPlay={(event) => { if (event.currentTarget.currentTime < start || event.currentTarget.currentTime >= end) event.currentTarget.currentTime = start }} onTimeUpdate={(event) => { if (event.currentTarget.currentTime >= end) event.currentTarget.pause() }} className="aspect-video w-full bg-black object-contain" /> : <img src={imageUrl} alt={caption} loading="lazy" className="aspect-video w-full bg-black object-contain" />}
    <figcaption className="flex gap-2.5 px-3.5 py-3 text-xs leading-5 text-muted-foreground"><Camera className="mt-0.5 h-3.5 w-3.5 shrink-0 text-primary" /><span><strong className="font-medium text-foreground">{zh ? "视频片段" : "Video clip"}</strong> · {Math.floor(timeSec / 60)}:{String(Math.floor(timeSec % 60)).padStart(2, "0")} · F{frame} · {caption}</span></figcaption>
  </figure>
}

export function ChatPage() {
  const { t, language } = useLanguage()
  const zh = language === "zh"
  const [matches, setMatches] = useState<ChatMatch[]>([])
  const [providers, setProviders] = useState<ChatProviderInfo[]>([])
  const [jobId, setJobId] = useState("")
  const [provider, setProvider] = useState<ChatProvider>("agnes")
  const [messages, setMessages] = useState<DisplayMessage[]>([])
  const [input, setInput] = useState("")
  const [initialLoading, setInitialLoading] = useState(true)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [historyError, setHistoryError] = useState("")
  const [sending, setSending] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [retry, setRetry] = useState(0)
  const [showLatest, setShowLatest] = useState(false)
  const [copied, setCopied] = useState<number | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)
  const followRef = useRef(true)
  const controllerRef = useRef<AbortController | null>(null)
  const loading = initialLoading || historyLoading
  const match = matches.find((item) => item.job_id === jobId)
  const currentProvider = providers.find((item) => item.id === provider)
  const canSend = Boolean(jobId && currentProvider?.configured && !loading && !historyError && !sending && !clearing)
  const suggestions = zh ? [
    { title: "快速复盘", text: "概括这场比赛的回合、击球和对抗节奏。", icon: ChartNoAxesCombined },
    { title: "双方对比", text: "比较上下方球员的击球次数和移动负荷。", icon: MessageSquare },
    { title: "训练建议", text: "根据已有数据给出三条针对性训练建议。", icon: Sparkles },
  ] : [
    { title: "Match overview", text: "Summarize the rallies, hits and pace of this match.", icon: ChartNoAxesCombined },
    { title: "Player comparison", text: "Compare both players' hits and movement load.", icon: MessageSquare },
    { title: "Training ideas", text: "Give three training suggestions based on the data.", icon: Sparkles },
  ]

  useEffect(() => {
    const controller = new AbortController()
    void Promise.all([
      api.get<{ providers: ChatProviderInfo[] }>("/chat/config", { signal: controller.signal }),
      api.get<ChatMatch[]>("/chat/matches", { signal: controller.signal }),
    ]).then(([config, list]) => {
      if (controller.signal.aborted) return
      setProviders(config.data.providers)
      setMatches(list.data)
      setJobId(list.data[0]?.job_id ?? "")
      const savedProvider = localStorage.getItem("cm-chat-provider") as ChatProvider | null
      const preferred = config.data.providers.find((item) => item.id === savedProvider && item.configured)
        ?? config.data.providers.find((item) => item.configured)
      setProvider(preferred?.id ?? "agnes")
    }).catch((error) => { if (!controller.signal.aborted) toast.error(errorText(error)) })
      .finally(() => { if (!controller.signal.aborted) setInitialLoading(false) })
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    setMessages([])
    setHistoryError("")
    followRef.current = true
    setShowLatest(false)
    if (!jobId) return
    setHistoryLoading(true)
    void api.get<ChatHistory>(`/chat/${jobId}/history`, { params: { provider }, signal: controller.signal })
      .then(({ data }) => { if (!controller.signal.aborted) setMessages(data.messages) })
      .catch((error) => { if (!controller.signal.aborted) setHistoryError(errorText(error)) })
      .finally(() => { if (!controller.signal.aborted) setHistoryLoading(false) })
    return () => controller.abort()
  }, [jobId, provider, retry])

  useEffect(() => () => controllerRef.current?.abort(), [])

  useEffect(() => {
    const panel = scrollRef.current
    if (panel && followRef.current) panel.scrollTop = panel.scrollHeight
  }, [messages, sending, loading])

  function scrollToLatest() {
    followRef.current = true
    setShowLatest(false)
    const panel = scrollRef.current
    if (panel) panel.scrollTop = panel.scrollHeight
  }

  async function send(content = input) {
    const text = content.trim()
    if (!text || !canSend || controllerRef.current) return
    const controller = new AbortController()
    controllerRef.current = controller
    const user: DisplayMessage = { role: "user", content: text, created_at: Date.now() / 1000 }
    const assistant: DisplayMessage = { role: "assistant", content: "", created_at: user.created_at + 0.001 }
    followRef.current = true
    setShowLatest(false)
    setMessages((items) => [...items, user, assistant])
    setInput("")
    setSending(true)
    try {
      const finalMessage = await streamReply(jobId, provider, text, controller.signal, (chunk) => {
        if (controller.signal.aborted) return
        setMessages((items) => items.map((item) => item.created_at === assistant.created_at ? { ...item, content: item.content + chunk } : item))
      })
      if (!controller.signal.aborted) setMessages((items) => items.map((item) => item.created_at === assistant.created_at ? finalMessage : item))
    } catch (error) {
      setMessages((items) => items.map((item) => item.created_at === assistant.created_at ? { ...item, incomplete: true } : item))
      if (!controller.signal.aborted) toast.error(errorText(error))
      setInput(text)
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null
      setSending(false)
    }
  }

  async function clear() {
    if (!jobId || sending || clearing) return
    setClearing(true)
    try {
      await api.delete(`/chat/${jobId}/history`, { params: { provider } })
      setMessages([])
      setInput("")
      setShowLatest(false)
      followRef.current = true
    } catch (error) { toast.error(errorText(error)) }
    finally { setClearing(false) }
  }

  return (
    <div className="h-full min-h-0 bg-background">
      <section className="relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden bg-background">
        <header className="pointer-events-none absolute inset-x-0 top-0 z-20 flex items-start justify-between p-3 sm:p-4">
          <div className="pointer-events-auto ml-12 min-w-0 lg:ml-0">
            <Select value={jobId} onValueChange={setJobId} disabled={initialLoading || sending || clearing}>
              <SelectTrigger className="h-10 w-[190px] rounded-xl border bg-background/90 text-xs shadow-lg backdrop-blur-xl sm:w-[250px]" aria-label={t("chat.match")}><SelectValue placeholder={loading ? t("common.loading") : t("chat.noMatches")} /></SelectTrigger>
              <SelectContent>{matches.map((item) => <SelectItem key={item.job_id} value={item.job_id}>{item.name} · {item.job_id.slice(0, 6)}</SelectItem>)}</SelectContent>
            </Select>
          </div>
          <Button variant="outline" size="icon" className="pointer-events-auto h-10 w-10 rounded-xl bg-background/90 text-muted-foreground shadow-lg backdrop-blur-xl" aria-label={t("chat.clear")} title={t("chat.clear")} onClick={() => void clear()} disabled={!messages.length || sending || loading || clearing}><RotateCcw className="h-3.5 w-3.5" /></Button>
        </header>

        <div className="relative flex min-h-0 flex-1 flex-col">
          <div ref={scrollRef} data-testid="chat-messages" role="log" aria-label={zh ? "对局分析消息" : "Match messages"} aria-live="off" className="chat-scroll min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 pb-36 pt-20 sm:px-8" onScroll={() => {
            const panel = scrollRef.current
            if (!panel) return
            const nearBottom = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 64
            followRef.current = nearBottom
            setShowLatest(!nearBottom)
          }}>
            <div className="mx-auto flex min-h-full w-full max-w-[780px] flex-col">
              {loading && <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground"><Loader2 className="mr-2 h-4 w-4 animate-spin" />{t("common.loading")}</div>}
              {!loading && historyError && <div className="m-auto space-y-3 text-center text-sm"><p className="text-destructive">{historyError}</p><Button variant="outline" onClick={() => setRetry((value) => value + 1)}>{zh ? "重新加载" : "Retry"}</Button></div>}
              {!loading && !historyError && !messages.length && <div className="my-auto py-5 sm:py-10">
                <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-2xl border border-primary/15 bg-primary/5 text-primary"><Sparkles className="h-6 w-6" /></div>
                <p className="mb-2 text-[11px] font-medium tracking-[0.16em] text-primary">{zh ? "每一场，都有新的发现" : "EVERY MATCH TELLS A STORY"}</p>
                <h3 className="text-2xl font-semibold leading-tight tracking-tight sm:text-3xl">{zh ? "想从这场对局中了解什么？" : "What can this match tell you?"}</h3>
                <p className="mt-3 max-w-lg text-sm leading-6 text-muted-foreground">{jobId ? (zh ? "聊聊比赛节奏、双方表现，或找到下一步的训练方向。" : "Explore the pace, compare performances, or find your next training focus.") : t("chat.noMatchesHelp")}</p>
                <div className="mt-7 grid gap-2 xl:grid-cols-3">{suggestions.map((item) => <button key={item.title} onClick={() => void send(item.text)} disabled={!canSend} className="group rounded-xl border bg-muted/15 p-4 text-left transition-colors hover:border-primary/30 hover:bg-primary/5 disabled:opacity-45"><span className="mb-2 flex items-center justify-between"><item.icon className="h-4 w-4 text-primary" /><ArrowUpRight className="h-3.5 w-3.5 text-muted-foreground transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" /></span><span className="block text-sm font-medium">{item.title}</span><span className="mt-1 block text-xs leading-5 text-muted-foreground">{item.text}</span></button>)}</div>
              </div>}
              {!loading && messages.map((item, index) => <article key={`${item.created_at}-${index}`} className={cn("mb-7", item.role === "user" ? "ml-auto max-w-[88%] sm:max-w-[78%]" : "w-full")}>
                {item.role === "user" ? <div className="whitespace-pre-wrap break-words rounded-2xl rounded-tr-md bg-muted/75 px-4 py-3 text-sm leading-7 [overflow-wrap:anywhere]">{item.content}</div> : <>
                  <div className="mb-3 flex items-center gap-2 text-xs font-medium"><span className="flex h-6 w-6 items-center justify-center rounded-lg bg-primary/10 text-primary"><Sparkles className="h-3.5 w-3.5" /></span>{zh ? "对局分析" : "Match analyst"}<span className="font-normal text-muted-foreground">· {currentProvider?.label}</span></div>
                  {item.content ? <Answer content={item.content} /> : !item.incomplete && <div className="flex items-center gap-2 py-1 text-sm text-muted-foreground"><Loader2 className="h-3.5 w-3.5 animate-spin" />{zh ? "正在阅读比赛数据…" : "Reading match data…"}</div>}
                  {item.frames?.map((frame) => <FrameClip key={`${frame.frame}-${frame.image_url}`} videoUrl={toFullUrl(match?.video_url)} imageUrl={toFullUrl(frame.image_url) ?? ""} timeSec={frame.time_sec} caption={frame.caption} frame={frame.frame} zh={zh} />)}
                  {sending && index === messages.length - 1 && item.content && <span className="mt-2 inline-block h-3 w-1 animate-pulse rounded bg-primary" />}
                  {item.incomplete && <p className="mt-3 text-xs text-amber-600">{zh ? "本次回复未完成，内容尚未保存。可重新发送下方问题。" : "Reply incomplete and not saved. Resend your question below."}</p>}
                  {item.content && !sending && <button className="mt-3 flex items-center gap-1.5 rounded py-1 text-[11px] text-muted-foreground hover:text-foreground" onClick={() => { void navigator.clipboard.writeText(cleanInternalReferences(item.content)).then(() => setCopied(index)).catch(() => toast.error(zh ? "复制失败" : "Copy failed")) }}>{copied === index ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}{zh ? (copied === index ? "已复制" : "复制") : (copied === index ? "Copied" : "Copy")}</button>}
                </>}
              </article>)}
            </div>
          </div>
          {showLatest && <button onClick={scrollToLatest} aria-label={zh ? "回到最新消息" : "Jump to latest"} className="absolute bottom-3 left-1/2 flex h-8 w-8 -translate-x-1/2 items-center justify-center rounded-full border bg-background shadow-md"><ArrowDown className="h-4 w-4" /></button>}
        </div>

        <footer className="pointer-events-none absolute inset-x-0 bottom-0 z-20 px-3 pb-4 sm:px-6 sm:pb-5">
          <div className="pointer-events-auto mx-auto max-w-[800px]">
            {!initialLoading && !currentProvider?.configured && <p className="mb-2 px-1 text-xs text-amber-600">{t("chat.configureKey")}</p>}
            <form onSubmit={(event) => { event.preventDefault(); void send() }} className="flex items-end gap-2 rounded-[26px] border bg-background/95 p-2 pl-4 shadow-xl backdrop-blur-xl">
              <textarea aria-label={zh ? "输入对局问题" : "Your match question"} value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send() } }} rows={1} maxLength={4000} disabled={!jobId || sending || loading} placeholder={zh ? "问问这场比赛…" : "Ask about this match…"} className="chat-scroll block max-h-32 min-h-10 flex-1 resize-none border-0 bg-transparent py-2 text-sm leading-6 outline-none placeholder:text-muted-foreground/70 disabled:opacity-50" />
              {sending ? <Button type="button" size="icon" className="h-10 w-10 shrink-0 rounded-full bg-blue-600 text-white hover:bg-blue-700" onClick={() => controllerRef.current?.abort()} aria-label={zh ? "停止生成" : "Stop generating"} title={zh ? "停止生成" : "Stop generating"}><Square className="h-3 w-3 fill-current" /></Button> : <Button type="submit" size="icon" className="h-10 w-10 shrink-0 rounded-full bg-blue-600 text-white hover:bg-blue-700" disabled={!input.trim() || !canSend} aria-label={t("chat.send")}><ArrowUp className="h-4 w-4" /></Button>}
            </form>
          </div>
        </footer>
      </section>
    </div>
  )
}
