export interface AnalysisOptions {
  language: string
  pose_family: "yolo-pose" | "rtmpose" | "rtmo"
  pose_mode: string
  yolo_pose_model?: string
  ball_model?: string
  shuttle_model?: "yolo" | "tracknet" | "tracknet_v4" | "ensemble"
  audio: boolean
  show_skeletons: boolean
  show_player_trajectories: boolean
  show_court_trajectory: boolean
  show_shuttlecock_trajectory: boolean
  show_player_stats: boolean
  show_pose_roi: boolean
  visualize_positions: boolean
}

export interface AnalysisStartResponse {
  job_id: string
  status: string
}

export interface BatchJobStatus {
  job_id: string
  video_id: string
  filename: string
  status: JobStatus
  progress: number
  current_frame: number
  total_frames: number
  stage?: string | null
  error?: string | null
}

export interface BatchStatusResponse {
  batch_id: string
  status: "queued" | "processing" | "finished" | "completed" | "completed_with_errors"
  total: number
  finished: number
  completed: number
  failed: number
  progress: number
  current_index: number
  created_at?: number | null
  started_at?: number | null
  finished_at?: number | null
  jobs: BatchJobStatus[]
}

export type JobStatus =
  | "queued"
  | "processing"
  | "completed"
  | "failed"

export interface JobStatusResponse {
  job_id: string
  status: JobStatus
  progress: number
  current_frame: number
  total_frames: number
  stage?: string | null
  message?: string | null
  error?: string | null
  created_at?: number | null
  finished_at?: number | null
}

export interface JobResult {
  job_id: string
  status: JobStatus
  output_dir?: string | null
  video_url?: string | null
  metadata_url?: string | null
  detections_url?: string | null
  report_url?: string | null
  visualizations: string[]
  metadata?: Record<string, unknown> | null
  report?: MatchReport | null
  error?: string | null
}

export interface TimelinePlayer {
  court?: [number, number] | null
  speed_mps: number
  match_distance_m: number
  rally_distance_m: number
  match_average_speed_mps: number
  match_max_speed_mps: number
  rally_average_speed_mps: number
  rally_max_speed_mps: number
}

export interface TimelinePoint {
  frame: number
  time_sec: number
  rally_id?: number | null
  rally_hit_count: number
  rally_total_hits: number
  players: { upper: TimelinePlayer; lower: TimelinePlayer }
  shuttlecock: {
    image?: [number, number] | null
    visible: boolean
    sources: string[]
  }
}

export interface JobTimeline {
  job_id: string
  sample_hz: number
  duration_sec?: number | null
  total_frames?: number | null
  points: TimelinePoint[]
}

export interface RallySummary {
  rally_count: number
  total_hits: number
  hits_by_player: { upper: number; lower: number }
  average_hits_per_rally: number
  median_hits_per_rally: number
  longest_rally_id?: number | null
  longest_rally_hits: number
  active_rally_time_sec: number
  hits_per_active_minute: number
}

export interface RallyItem {
  rally_id: number
  start_frame: number
  end_frame: number
  start_sec: number
  end_sec: number
  duration_sec: number
  hit_count: number
  hits_by_player: { upper: number; lower: number }
  average_hit_interval_sec?: number | null
  confidence: number
  intensity: "low" | "medium" | "high"
  movement: Record<string, { distance_m: number; average_speed_mps: number; max_speed_mps: number }>
}

export interface MatchReport {
  schema_version: string
  algorithm: string
  title: string
  overview: string
  summary: RallySummary
  data_quality: {
    processed_records: number
    video_total_frames: number
    shuttle_observed_frames: number
    shuttle_coverage: number
    average_hit_confidence: number
  }
  players: Record<"upper" | "lower", {
    hit_count: number
    hit_share: number
    distance_m: number
    average_speed_mps: number
    max_speed_mps: number
  }>
  insights: Array<{ kind: string; title: string; message: string; rally_id?: number }>
  rallies: RallyItem[]
  limitations: string
}

export interface HistoryItem {
  job_id: string
  name: string
  status: string
  created_at?: number | null
  finished_at?: number | null
  duration_sec?: number | null
  video_url?: string | null
  metadata_url?: string | null
  error?: string | null
}

export interface SystemStatus {
  device: string
  gpu_available: boolean
  gpu_name?: string | null
  courtkeynet_weights: string
  courtkeynet_ready: boolean
  ball_model: string
  ball_model_ready: boolean
  yolo_pose_model: string
  yolo_pose_model_ready: boolean
  rtmpose_model_ready: boolean
  rtmo_model_ready: boolean
  tracknet_tracker_weights?: string
  tracknet_tracker_ready?: boolean
  tracknet_rectifier_weights?: string
  tracknet_rectifier_ready?: boolean
  tracknet_v4_weights?: string
  tracknet_v4_weights_ready?: boolean
  tracknet_v4_dependency_ready?: boolean
}
