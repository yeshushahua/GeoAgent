# GeoAgent Phase 0 实际验收记录

验收日期：2026-09-11。全部结果来自本机真实执行，不包含虚构的浏览器截图或模型能力。

## 1. 最终结论

**PARTIAL**。工程、GPU、pytest、smoke test、Gradio HTTP 交互均通过。
浏览器工具两次启动失败，错误为 `helper_unknown_error: setup refresh had errors`，
未完成浏览器图形界面目视验收，因此不宣称整体 PASS。

## 2. 项目目录

`D:\sht\DEMO\GeoAgent`

## 3. D 盘内容

backend（API/core/services/schemas + model/tool/agent 边界）、frontend、scripts、
26 项 tests、sample_data/images 小型原创合成 PNG、docs、README、MIT LICENSE、
.env/.env.example、.gitignore、pyproject.toml、依赖清单/锁定文件、.venv、.git。
没有下载模型或大型数据。Python/CUDA 安装依赖在允许的 .venv 中。

## 4. E 盘内容

models、cache/huggingface、datasets、outputs、checkpoints、temp 已建立并通过实际写入探针。
另有 cache/pip 安装缓存与 temp/gradio 上传预览缓存。
models、datasets、checkpoints 在验收时为空。

## 5. Python 环境

- Python：3.12.14（项目 .venv；基于 Codex bundled Python 创建）
- PyTorch：2.10.0+cu128
- torchvision：0.25.0+cu128
- CUDA runtime：12.8
- NVIDIA driver：591.86；nvidia-smi 显示驱动支持 CUDA 13.1
- GPU：NVIDIA GeForce RTX 4090
- VRAM：23.99 GiB（25,756,696,576 bytes）
- 真实 CUDA tensor 运算：(ones(8) * 2).sum() == 16.0
- 未修改系统 Python、驱动或 CUDA Toolkit；未配置其他 GPU。

## 6. FastAPI

已真实启动，当前地址 http://127.0.0.1:8000。
health 与 system 均 HTTP 200，实际响应见文末。
启动日志包括 GeoAgent starting、Project root、Storage root、CUDA available: True、GPU 名称。

## 7. Gradio

地址：http://127.0.0.1:7860；页面 HTTP 200。
通过 gradio_client 的真实 HTTP 上传、队列事件、下载接口验证：

- 上传 480×300 原创小图。
- 返回 PNG 预览与原图大小和 RGB 像素完全一致。
- hello 消息正确进入聊天历史，并返回 Phase 1 占位回复。
- UI 状态函数真实请求 /health 和 /system，返回 Ready、CUDA=True、RTX 4090 与 E 盘路径。
- 实际停止 FastAPI 后，Gradio 仍可访问，显示 Backend unavailable；随后恢复后端并重验成功。
- 浏览器 GUI 上传控件、两栏显示和聊天气泡尚需人工目视确认。

已修复系统代理影响 localhost 启动的问题，只对 Gradio 进程增加本地地址代理排除项。
已将 Gradio 默认有损 WebP 改为无损 PNG。
第一次客户端验收曾触发 Hub 元数据/遥测请求（未下载模型），随后显式启用离线模式、关闭遥测；最终重验仅请求本机地址。

## 8. Storage

配置见文末。已实际验证 Hugging Face 库解析出的 HF_HOME 和 HF_HUB_CACHE 位于指定 E 盘。
HF_HUB_OFFLINE=true、HF_HUB_DISABLE_TELEMETRY=1；Phase 1 联网下载前修改 .env 的离线开关。
不提供 C/D 大资产回退。

## 9. Tests

```text
platform win32 -- Python 3.12.14, pytest-9.1.1, pluggy-1.6.0
collected 26 items
backend/tests/test_config.py ..........
backend/tests/test_frontend.py ..
backend/tests/test_health.py .
backend/tests/test_storage.py ....
backend/tests/test_system.py ..
backend/tests/test_tool_result.py .......
26 passed, 2 warnings in 0.22s
```

两条 warning 均来自依赖内部弃用提示：Starlette TestClient 的 httpx 路径，以及 anyio BlockingPortal 别名。
测试均通过，不屏蔽 warning，不额外引入当前不必需的 httpx2。
`pip check`：No broken requirements found.
CI 无 CUDA 分支已验证返回 HTTP 200、cuda_available=false 并记录警告。

## 10. Smoke Test

```text
Smoke test PASS: health, system, storage, HF cache, torch, RTX 4090 computation, ToolResult
Gradio HTTP test PASS: page 200, real backend RTX 4090 status, image upload/preview pixel equality, chat placeholder
Gradio unavailable-backend HTTP test PASS
```

## 11. Git

已 git init，未 stage、未 commit、未 push。待提交文件均小于 1 MiB。
.env、.venv、模型权重、大型数据和 cache 未进入 Git；没有添加密钥、token 或密码。
因仓库由沙箱账户初始化，为当前用户添加了仅此路径的 safe.directory 设置。
模型源码目录 backend/app/models 正常包含在待提交文件列表，根目录 /models 被忽略。
实际 git status 见文末。

## 12. Phase 0 PASS Checklist

- [x] 代码位于 D:\sht\DEMO\GeoAgent
- [x] 大型资产根目录位于 E:\sht\DEMO\GeoAgent
- [x] E 盘 models 目录
- [x] E 盘 HF cache 目录
- [x] E 盘 datasets 目录
- [x] E 盘 outputs 目录
- [x] E 盘 checkpoints 目录
- [x] Python 3.12 正常
- [x] .venv 正常
- [x] torch import 正常
- [x] CUDA available=True
- [x] NVIDIA RTX 4090
- [x] FastAPI 启动
- [x] health HTTP 200
- [x] system HTTP 200
- [x] Gradio 启动
- [ ] 图片上传/显示的浏览器目视确认（真实上传/预览 HTTP 事件已通过）
- [x] Chat 占位功能（真实 HTTP 事件）
- [x] Gradio 读取真实 FastAPI 状态
- [x] ToolResult Schema
- [x] Logging
- [x] pytest 全部通过
- [x] smoke_test PASS
- [x] .env 未进入 Git
- [x] 模型权重未进入 Git
- [x] 大型数据未进入 Git
- [x] README 基础文档
- [x] git status 无异常大型或敏感文件

## 13. 未完成问题

仅待人工浏览器确认：打开 Gradio，上传 sample_data/images/foundation-demo.png，
确认 Input/Preview 两栏显示，输入 hello 查看用户/Agent 消息，点击 Refresh status 查看 RTX 4090。
服务保持运行供验收。当前监听进程：后端 41812，前端 31192（本次记录；重启会变化）。
可停止对应进程，或以后按 README 在两个终端启动并使用 Ctrl+C 停止。
不要将 HTTP 验收误称为已完成浏览器 GUI 测试。

## 实际 API 响应

health:
```json
{
  "status": "ok",
  "service": "geoagent",
  "version": "0.1.0"
}
```

system:
```json
{
  "python_version": "3.12.14",
  "torch_version": "2.10.0+cu128",
  "cuda_available": true,
  "cuda_version": "12.8",
  "gpu_name": "NVIDIA GeForce RTX 4090",
  "gpu_vram_gb": 23.99,
  "project_root": "D:\\sht\\DEMO\\GeoAgent",
  "storage_root": "E:\\sht\\DEMO\\GeoAgent"
}
```

## 实际存储配置

```json
{
  "MODEL_DIR": "E:\\sht\\DEMO\\GeoAgent\\models",
  "HF_HOME": "E:\\sht\\DEMO\\GeoAgent\\cache\\huggingface",
  "DATASET_DIR": "E:\\sht\\DEMO\\GeoAgent\\datasets",
  "OUTPUT_DIR": "E:\\sht\\DEMO\\GeoAgent\\outputs",
  "CHECKPOINT_DIR": "E:\\sht\\DEMO\\GeoAgent\\checkpoints",
  "TEMP_DIR": "E:\\sht\\DEMO\\GeoAgent\\temp"
}
```

## 实际 Git 状态

```text
?? .env.example
?? .gitignore
?? LICENSE
?? README.md
?? backend/
?? docs/
?? frontend/
?? pyproject.toml
?? requirements-gpu.txt
?? requirements-lock.txt
?? requirements.txt
?? sample_data/
?? scripts/
```
