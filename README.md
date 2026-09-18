# CourtMind

> 面向羽毛球比赛视频的智能分析平台：从球场标定、人体姿态与羽毛球轨迹检测，到 Rally 自动切分、击球统计、比赛报告和基于比赛数据的 AI 对局问答。

**简体中文** | [English](README_EN.md)

[![CourtMind AI 对局分析：回答内嵌击球片段](docs/images/chat-poster.jpg)](docs/images/pay.mp4)

*点击封面播放演示视频（AI 对局分析，回答内嵌击球片段）。*

## 项目简介

CourtMind 采用 React + FastAPI 的前后端架构。用户上传比赛视频后，系统会自动识别球场、跟踪双方球员和羽毛球，并把视觉算法结果转化为可读的比赛指标、训练建议和可回看的关键片段。

AI 对局分析支持流式回答。回答会优先引用当前比赛的结构化分析数据；在适合进行动作或站位复盘的问题中，还可以在聊天消息内直接展示对应时间点的视频片段。

## 主要功能

- CourtKeyNet 自动检测球场四角，支持人工微调。
- TrackNetV4 Type B、TrackNetV3、YOLO11 三种羽毛球检测模型。
- 三模型时序融合：多候选关联、置信度与运动一致性加权，并对歧义结果主动拒判。
- 球员检测、跟踪与人体姿态骨架绘制。
- 羽毛球轨迹、球员轨迹和俯视角球场可视化。
- Rally 自动切分、击球次数统计、移动距离与节奏分析。
- 单视频分析与最多 50 个视频的顺序批量分析。
- 自动生成面向用户的比赛报告与针对性训练建议。
- Agnes、DeepSeek 对局问答，支持 SSE 流式输出和消息内视频片段。
- 分析历史、明暗主题、模型与 LLM 配置管理。

> 算法给出的击球、Rally、站位和移动指标属于计算机视觉估计，不等同于裁判计分或正式比赛结论。低置信度、遮挡和低检测覆盖率会在报告中明确提示。

## 界面演示

[![CourtMind 分析结果页](docs/images/analysis-poster.jpg)](docs/images/analysis.mp4)

*分析结果页：标注视频与右侧实时指标随播放进度同步。*

## 技术栈

| 层级 | 技术 |
| --- | --- |
| 前端 | React 18、TypeScript、Vite、Tailwind CSS、Radix UI |
| 后端 | Python 3.10、FastAPI、Uvicorn |
| 视觉推理 | PyTorch、Keras 3、Ultralytics、ONNX Runtime、RTMLib、OpenCV |
| 核心模型 | CourtKeyNet、TrackNetV4、TrackNetV3、YOLO11、RTMPose / RTMO |
| LLM | Agnes、DeepSeek（OpenAI-compatible Chat Completions） |

## 环境要求

- Windows 10/11
- [Miniconda 或 Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html)
- Node.js 18 或更高版本
- Python 3.10（推荐使用名为 `badminton` 的 conda 环境）
- NVIDIA GPU 为可选项；使用 CUDA 可以明显缩短长视频分析时间

本项目当前验证环境为 Python `3.10.21`，前后端默认地址分别是：

- Web：<http://127.0.0.1:5173>
- API：<http://127.0.0.1:8000>
- API 文档：<http://127.0.0.1:8000/docs>

## 模型权重

模型权重请从下列项目官网或官方发布页面下载。PyTorch 的 `.pt` 文件可能包含 pickle 数据，请只使用可信来源的权重。

| 模型 | 下载网站 |
| --- | --- |
| CourtKeyNet | [CourtKeyNet — Hugging Face](https://huggingface.co/Cracked-ANJ/CourtKeyNet) |
| YOLO11 羽毛球检测 | [Good-Badminton v0.1.0 Release](https://github.com/yo-WASSUP/Good-Badminton/releases/tag/v0.1.0) |
| YOLO11 人体姿态 | [Ultralytics Assets](https://github.com/ultralytics/assets/releases) |
| RTMPose / RTMO | [Good-Badminton Release](https://github.com/yo-WASSUP/Good-Badminton/releases/tag/v0.1.0) / [RTMLib 模型库](https://github.com/Tau-J/rtmlib) |
| TrackNetV3 / InpaintNet | [TrackNetV3 官方权重（Google Drive）](https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view?usp=sharing) |
| TrackNetV4 Type B | [TrackNetV4 官方结果与权重页面](https://github.com/TrackNetV4/TrackNetV4/blob/main/docs/RESULT.md) |

下载完成后，可在“设置”页面检查各模型是否已经就绪。

## 安装

以下命令均在 Windows CMD 中执行。

### 1. 创建 conda 环境

```bat
conda create -n badminton python=3.10 -y
conda activate badminton
```

### 2. 安装后端依赖

```bat
cd CourtMind
python -m pip install -r requirements.txt
```

`requirements.txt` 使用 CUDA 12.4 对应的 PyTorch wheel。若本机驱动或 CUDA 环境不同，请先根据 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 选择匹配版本，再安装其余依赖。

### 3. 安装前端依赖

```bat
cd frontend
npm install
cd ..
```

### 4. 配置 LLM（可选）

```bat
copy .env.example .env
```

编辑 `.env`，填入自己申请的密钥：

```env
AGNES_API_KEY=your-agnes-api-key
AGNES_MODEL=agnes-2.5-flash
AGNES_CHAT_URL=https://apihub.agnes-ai.com/v1/chat/completions

DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_CHAT_URL=https://api.deepseek.com/chat/completions
```

`.env` 已被 Git 忽略。不要把真实密钥写入 `.env.example`、README、前端源码或提交记录。

## 一键启动与停止

### CMD 推荐方式

在项目根目录直接执行：

```bat
start.bat
```

脚本会在后台启动 FastAPI 和 Vite，并等待两个服务通过健康检查。启动完成后访问 <http://127.0.0.1:5173>。

停止服务：

```bat
stop.bat
```

> 不要输入 `bash start.bat` 或 `bash stop.bat`。`.bat` 是 Windows CMD 脚本，前面加 `bash` 会出现 `@echo: command not found`、`Exec format error` 等错误。

### PowerShell

```powershell
.\start.ps1
.\stop.ps1
```

`start.sh` 只用于从 Git Bash / WSL 转交启动；在 CMD 中优先使用 `start.bat`。运行日志保存在 `logs/`。

### 手动启动（排错用）

后端窗口：

```bat
conda activate badminton
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

前端窗口：

```bat
cd frontend
npm run dev
```

## 使用流程

1. 在“新建分析”上传单打比赛视频，并为本次分析命名。
2. 自动检测球场；如角点有偏差，在画面中拖动修正。
3. 在“设置”中选择姿态模型、羽毛球模型和 AI 服务商。
4. 开始分析，等待任务完成。
5. 在结果页同步播放分析视频与右侧指标，查看统计和比赛报告。
6. 进入“对局分析”，选择已完成的比赛，针对移动、击球、战术和训练提出问题。

批量任务请从“批量分析”进入。后端会按顺序执行视频，避免同时加载多个三模型融合任务而耗尽显存或内存；单个视频失败不会中断其余队列。

## 羽毛球检测模式

| 模式 | 说明 |
| --- | --- |
| `ensemble` | TrackNetV4 + TrackNetV3 + YOLO11 时序融合；强调精度，计算量最大 |
| `tracknet_v4` | TrackNetV4 Type B，使用运动注意力融合 |
| `tracknet` | TrackNetV3 跟踪器 + InpaintNet 轨迹修复 |
| `yolo` | YOLO11 单帧检测，速度较快，保留为兼容模式 |

融合模式不是简单平均坐标。系统保留各模型的多个候选点，根据观测置信度、模型可靠度、预测轨迹的不确定性和速度/方向一致性进行关联；只有连续帧得到支持的候选才会被确认，对互相冲突且无法可靠区分的结果会拒绝输出。更多实现说明见 [docs/temporal-fusion.md](docs/temporal-fusion.md)。

## 输出结果

每次任务写入 `outputs/<job_id>/`，主要包含：

| 文件 | 内容 |
| --- | --- |
| `detect_<原视频名>.mp4` | 人体姿态骨架、球员/羽毛球轨迹和俯视角球场 |
| `metadata.json` | 视频、模型、球场、轨迹与统计元数据 |
| `match_report.json` | Rally、击球、移动负荷、关键时刻和训练建议 |
| `chat_history.json` | 当前比赛按 LLM 服务商保存的对话历史 |
| `chat_media/` | AI 回答引用的帧图或短视频片段 |

数值统计不会烧录在输出视频旁边；结果页会根据视频当前播放时间同步更新对应数据。

## 项目结构

```text
CourtMind/
├─ backend/                       # FastAPI 服务
│  ├─ api/                        # 视频、球场、任务、批量、聊天等接口
│  ├─ core/                       # 配置、环境变量与路径
│  ├─ schemas/                    # Pydantic 请求/响应模型
│  ├─ services/                   # 分析编排、任务队列、LLM、视频时间线
│  └─ main.py                     # API 入口、CORS、静态文件挂载
├─ badminton_analysis/            # 项目内可独立维护的算法代码
│  ├─ analysis/                   # 统计与分析逻辑
│  ├─ court/                      # 球场标定与坐标变换
│  ├─ detection/                  # 姿态与基础检测器
│  ├─ events/                     # Rally 与击球事件分析
│  ├─ models/                     # TrackNetV3/V4、YOLO11 本地模型实现
│  ├─ shuttlecock/                # 单模型适配与三模型时序融合
│  ├─ tracking/                   # 球员跟踪
│  └─ visualization/              # 骨架、轨迹、俯视角绘制
├─ frontend/                      # React 18 + TypeScript + Vite
│  └─ src/
│     ├─ components/              # 布局、分析流程与通用 UI
│     ├─ hooks/                   # 分析、轮询、球场检测等 hooks
│     ├─ lib/                     # API、流式聊天、坐标与工具函数
│     ├─ pages/                   # 仪表盘、分析、结果、聊天、历史、设置
│     ├─ router/                  # 前端路由
│     └─ types/                   # TypeScript 类型
├─ docs/                          # 设计与算法文档、README 截图与演示视频
├─ weights/                       # 模型权重（大文件通常不提交 Git）
├─ uploads/                       # 上传视频与球场状态
├─ outputs/                       # 每个分析任务的结果
├─ logs/                          # 前后端启动日志
├─ tests/                         # 自动化测试
├─ requirements.txt              # Python 依赖
├─ start.bat / start.ps1 / start.sh
└─ stop.bat / stop.ps1
```

TrackNetV3、TrackNetV4、YOLO11 的项目内代码位于 `badminton_analysis/models/`；相邻目录中的 `BadmintonTrackNet`、`TrackNetV4` 等原始模型项目保持独立，不会被本项目删除或覆盖。

## 常见问题

### 端口 8000 已被占用

先在项目根目录执行：

```bat
stop.bat
start.bat
```

如果仍提示占用，可查看 `logs/backend-*.error.log`，并用 `netstat -ano | findstr :8000` 确认占用进程。

### 前端启动超时

检查最新的 `logs/frontend-*.error.log`。常见原因是尚未执行 `frontend\npm install`，或 5173 端口被其他 Vite 进程占用。

### 三模型融合内存不足

批量任务会顺序运行，但高分辨率视频仍可能占用较多 RAM/显存。可先关闭其他 GPU 程序、降低输入视频分辨率，或在设置中改用单个 TrackNetV4 / TrackNetV3 模型。

### AI 对话不可用

确认 `.env` 中至少配置了一个有效 API Key，修改后重启后端；然后在“设置”中选择对应服务商。AI 回答只依据所选比赛已经生成的分析数据，不会把检测结果当作真实比分或胜负。

## 致谢

CourtMind 的实现受益于以下优秀的开源项目、论文实现和模型生态，感谢所有作者与维护者：

- [Good-Badminton](https://github.com/yo-WASSUP/Good-Badminton)：原始羽毛球视频分析项目与部分模型资产。
- [TrackNetV3](https://github.com/qaz812345/TrackNetV3)：羽毛球轨迹预测与 InpaintNet 轨迹修复。
- [TrackNetV4](https://github.com/TrackNetV4/TrackNetV4)：运动注意力与 Type A / Type B 融合框架。
- [BadmintonTrackNet](https://github.com/ZSHYC/BadmintonTrackNet)：TrackNetV3 工程化、推理与事件分析参考。
- [Ultralytics](https://github.com/ultralytics/ultralytics)：YOLO11 检测与姿态推理生态。
- [RTMLib](https://github.com/Tau-J/rtmlib)：轻量化 RTMPose / RTMO 推理接口。
- [MMPose](https://github.com/open-mmlab/mmpose)：OpenMMLab 人体姿态模型与预训练资产。
- [CourtKeyNet](https://huggingface.co/Cracked-ANJ/CourtKeyNet)：羽毛球场关键点检测权重。
- [FastAPI](https://github.com/fastapi/fastapi)、[React](https://github.com/facebook/react) 与 [Vite](https://github.com/vitejs/vite)：本项目的 Web 应用基础设施。

使用、发布或再分发上述代码与权重时，请同时遵守各上游项目自己的许可证和模型条款。

## 许可证

本仓库代码按 [Apache License 2.0](LICENSE) 发布。第三方源码、模型权重、数据集和服务 API 不因集成到本项目而改变其原始许可，请以各自上游声明为准。
