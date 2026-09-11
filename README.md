# GeoAgent

**Multimodal AI Agent for Visual & Geospatial Analysis**  
多模态视觉与空间智能分析 Agent · 当前阶段：**Phase 0 — Project Foundation**

本阶段提供 Python 工程、CUDA 检查、FastAPI、Gradio、路径配置、日志、
统一 ToolResult 和自动化测试。聊天仅返回明确的占位回复，图片仅预览。
**Qwen3-VL-4B-Instruct 将在 Phase 1 接入；当前没有下载、加载或运行模型。**
检测、分割、遥感、GeoTIFF、Agent 推理和微调均尚未实现。

## 架构

```text
Gradio ── HTTP ── FastAPI /api/v1
                     │
                Core / Services / Schemas

未来：Agent → Tool → Model / Service
```

- `backend/app/core/`：配置与统一 logging。
- `backend/app/services/`：存储目录准备、系统状态检测。
- `backend/app/schemas/`：ToolResult、Artifact、ToolError。
- `backend/app/models/`：预留模型加载、生命周期和推理封装。
- `backend/app/tools/`：预留 Agent 可调用工具，统一返回 ToolResult。
- `backend/app/agent/`：预留状态、规划、工具编排。
- `frontend/app.py`：Gradio 上传、预览、聊天占位和真实后端状态。

Agent 只能调用 Tool，不能直接执行 PyTorch / CV / GIS 算法。
本阶段仅使用单张 RTX 4090，不配置其他 GPU 或分布式运行。

## 环境与安装（Windows PowerShell）

需要 Python **3.12** 和可用的 NVIDIA RTX 4090 驱动。先检查：

```powershell
Set-Location D:\sht\DEMO\GeoAgent
py --version
py -0p
nvidia-smi
py -3.12 -m venv .venv
Copy-Item .env.example .env
```

已经有 `.env` 时不要覆盖；其他电脑只需要修改 `.env`，不需要修改 Python 源码。
当前机器使用 Codex 自带 Python 3.12.14 创建 `.venv`，未更改系统 Python；
该虚拟环境依赖原解释器，若原解释器移除，请安装 Python 3.12 后重建 `.venv`。
不需要激活环境，下列命令直接指定项目解释器。

确认 `.env` 中目录正确后，先将安装缓存和临时下载放到配置的存储盘：

```powershell
# 与 .env 中 STORAGE_ROOT / TEMP_DIR 保持一致。
$assetRoot = 'E:\sht\DEMO\GeoAgent'
New-Item -ItemType Directory -Force -Path "$assetRoot\temp", "$assetRoot\cache\pip"
$env:TEMP = "$assetRoot\temp"
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = "$assetRoot\cache\pip"
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

CUDA wheel 固定为 PyTorch 2.10.0 + CUDA 12.8、torchvision 0.25.0。
安装来源为 [PyTorch 官方 CUDA wheel 仓库](https://download.pytorch.org/whl/cu128)。
不修改系统 CUDA Toolkit。`nvidia-smi` 的 CUDA 版本是驱动支持版本，
系统 API 的 `cuda_version` 是 PyTorch 自带 runtime，二者可以不同。
`requirements-lock.txt` 记录本次实际验收版本；重建相同环境时，在 GPU 安装后
用 `pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128`。

## 启动

在两个 PowerShell 窗口中进入项目根目录，分别运行：

```powershell
.\.venv\Scripts\python.exe -m backend.app.main
```

```powershell
.\.venv\Scripts\python.exe -m frontend.app
```

- Gradio：http://127.0.0.1:7860
- Health：http://127.0.0.1:8000/api/v1/health
- System：http://127.0.0.1:8000/api/v1/system
- API 文档：http://127.0.0.1:8000/docs

地址和端口读取 `.env`。默认仅本机访问；Ctrl+C 停止服务。
Gradio 启动时和点击 Refresh status 时真实请求两个 API。
后端未启动时显示 **Backend unavailable**；CUDA 不可用时 API 仍返回 200，
日志输出警告，UI 提示 GPU 需要检查。Ready 要求后端可用且 GPU 为 RTX 4090。
上传 `sample_data/images/foundation-demo.png` 可检查两栏图片显示；
输入 hello 后显示用户消息和 Phase 1 占位回复。

## D / E 磁盘设计

| 配置 | 本机目录 | 用途 |
|---|---|---|
| PROJECT_ROOT | D:/sht/DEMO/GeoAgent | 源码、文档、测试、Git、.venv、小图 |
| STORAGE_ROOT | E:/sht/DEMO/GeoAgent | 大型资产根目录 |
| MODEL_DIR | E:/sht/DEMO/GeoAgent/models | 正式权重 |
| HF_HOME | E:/sht/DEMO/GeoAgent/cache/huggingface | Hugging Face 缓存 |
| DATASET_DIR | E:/sht/DEMO/GeoAgent/datasets | 数据集 |
| OUTPUT_DIR | E:/sht/DEMO/GeoAgent/outputs | 推理、可视化、评测产物 |
| CHECKPOINT_DIR | E:/sht/DEMO/GeoAgent/checkpoints | adapter / checkpoint |
| TEMP_DIR | E:/sht/DEMO/GeoAgent/temp | 临时处理和 Gradio 上传缓存 |

启动校验绝对路径、项目与存储目录隔离、Windows 不同盘符、资产目录归属，
并创建必要目录、实际测试写权限。盘符缺失、目录不可写或越界会明确报错，
**绝不会静默回退 C/D 盘**。HF_HOME 及 Hub/Assets/Datasets 子缓存环境变量
在 UI/HF 库导入之前设置。Phase 0 默认 HF_HUB_OFFLINE=true，并关闭 Hub 遥测，不触发模型下载。Phase 1 下载前需在 .env 改为 false。
`.venv` 中的 PyTorch/CUDA 依赖属于允许的项目 Python 环境；模型权重不在 D 盘。

## Tests 与验收

```powershell
.\.venv\Scripts\python.exe -m pytest
# 先启动 FastAPI，再执行本机 GPU 验收：
.\.venv\Scripts\python.exe -m scripts.smoke_test
# FastAPI 与 Gradio 均启动后，验证上传、预览、聊天和状态：
.\.venv\Scripts\python.exe -m scripts.verify_ui
```

pytest 使用独立临时目录、包含无 CUDA 分支，不要求 CI 有 GPU 或 E 盘。
路径校验测试独立验证 Windows 跨盘策略。smoke test 则要求 Python 3.12、
本机与后端都检测到 RTX 4090，并执行一次很小的 CUDA tensor 运算。
实际 HTTP、Gradio 交互和环境验收记录见 `docs/phase0-validation.md`。

## Git 与安全边界

`.env`、`.venv`、缓存、产物、checkpoint、常见权重及大型 TIFF 均已忽略。
禁止提交 API key、token、password、模型权重、大型数据和缓存。
小型 demo 数据若将来需要例外，必须明确审查后再加入。
仓库仅初始化，不自动 commit 或 push。

## 下一阶段

Phase 1 将实现 Qwen3-VL-4B-Instruct → Model Manager → RTX 4090 →
Image + Prompt → Multimodal inference → Structured result。
当前只保留分层边界，不预装 LangGraph、YOLO、SAM、GIS 或微调依赖。
