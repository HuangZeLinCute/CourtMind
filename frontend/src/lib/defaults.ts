/** Frontend mirror of backend/core/config.py DEFAULT_OPTIONS. */
import type { AnalysisOptions } from "@/types"

export const DEFAULT_OPTIONS: AnalysisOptions = {
  language: "zh",
  pose_family: "yolo-pose",
  pose_mode: "balanced",
  yolo_pose_model: "weights/yolo11n-pose.pt",
  ball_model: "weights/yolo11s-ball.pt",
  shuttle_model: "tracknet",
  audio: true,
  show_skeletons: true,
  show_player_trajectories: true,
  show_court_trajectory: true,
  show_shuttlecock_trajectory: true,
  show_player_stats: false,
  show_pose_roi: false,
  visualize_positions: true,
}
