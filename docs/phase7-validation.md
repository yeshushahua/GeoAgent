# Phase 7 Validation — Remote Sensing Extension

## 1. Goal

Phase 7 extends the existing VisionAgent with foundational remote-sensing raster support. A user can upload a GeoTIFF, inspect structured geospatial metadata, generate a display preview, crop a pixel window without losing georeferencing, calculate raw-value band statistics, and send a generated preview to the existing image understanding tools.

Phase 7 does not add a separate agent or any new AI model.

## 2. Architecture

The existing path remains `Gradio -> FastAPI -> VisionAgent -> WorkflowController -> ToolRegistry -> ToolExecutor`. Four raster tools use rasterio behind the same executor and ToolResult contract. WorkflowController resolves raster sources independently from image analysis sources, and ResultAggregator publishes a structured raster summary.

Raster datasets are opened only inside a context manager. Workflow artifacts store paths and serializable metadata; they never retain an open rasterio dataset handle.

## 3. Raster Artifact

The artifact graph adds three types:

- `raster`: uploaded GeoTIFF and initial `raster_source`
- `raster_crop`: georeferenced GeoTIFF derived from a raster parent
- `raster_preview`: downsampled PNG visualization derived from a raster parent

A preview updates `active_image` so image tools can consume it. It does not replace `active_raster`. A crop updates `active_raster`. Raster tools reject a preview as a raster source.

## 4. Raster Tools

- `inspect_raster(raster_path)` reads metadata without reading pixel arrays.
- `raster_preview(raster_path, bands, stretch, lower_percentile, upper_percentile, max_size, resampling)` renders one or three 1-based bands as PNG.
- `crop_raster(raster_path, region | row_start, row_end, col_start, col_end)` performs strict pixel-window cropping. Named halves and quadrants are derived from actual width and height.
- `raster_statistics(raster_path, bands)` calculates per-band min, max, mean, population standard deviation, valid count, and NoData count.

All calls use registry-generated Pydantic schemas and pass through ToolExecutor. Raster failures use structured codes including `INVALID_RASTER`, `RASTER_OPEN_FAILED`, `BAND_NOT_FOUND`, `INVALID_WINDOW`, `EMPTY_WINDOW`, `UNSUPPORTED_RASTER_DTYPE`, and `PREVIEW_FAILED`.

## 5. GeoTIFF Metadata

The real RTX workflow used a 256 × 192, three-band `uint8` GeoTIFF. `inspect_raster` returned:

- Driver: GTiff
- CRS: EPSG:32647
- Transform: `[10, 0, 500000, 0, -10, 2000000]`
- Resolution: 10 × 10
- Bounds: left 500000, bottom 1998080, right 502560, top 2000000
- NoData: 0
- Band descriptions: Red, Green, Blue

Tests also cover a valid GeoTIFF with no CRS and no NoData value.

## 6. Preview

Preview uses `rasterio.read(..., out_shape=...)` and limits the longest edge to at most 2048 pixels. The default visualization is bands `[1, 2, 3]` for rasters with at least three bands and band `[1]` for a single-band raster. A two-band raster requires an explicit valid display selection.

The default stretch is the per-band 2nd–98th percentile; min-max is also available. Nearest and bilinear resampling are supported, with bilinear as the visual-preview default. Masked values, NoData, NaN, and Inf do not determine stretch limits.

## 7. Crop and Transform

Pixel windows use inclusive `row_start`/`col_start` and exclusive `row_end`/`col_end`. Validation is strict: empty and out-of-bounds windows return structured errors instead of silent clipping.

In the real non-square Case D, the named top-left quarter of 256 × 192 produced 128 × 96. CRS stayed EPSG:32647, resolution stayed 10 × 10, transform remained `[10, 0, 500000, 0, -10, 2000000]` because the window begins at the original origin, and bounds became left 500000, bottom 1999040, right 501280, top 2000000. Other windows use `src.window_transform(window)` to shift the affine origin correctly.

## 8. Statistics

Statistics operate on original raster values, independently from preview stretching. Each requested band is read through `src.block_windows(band)`. Raster masks, declared NoData, NaN, and Inf are excluded.

Real cropped-raster results included:

- Red: min 35, max 185, mean 85.4618, std 52.9236, valid 11564, NoData 724
- Green: min 95, max 185, mean 130.5777, std 26.2530, valid 11564, NoData 724
- Blue: min 65, max 190, mean 111.1363, std 57.6255, valid 11564, NoData 724

## 9. Large Raster Handling

The synthetic large-raster test creates a tiled 4096 × 4096 GeoTIFF dynamically. Metadata inspection does not read pixels. A preview requested at 256 pixels produces a 256 × 256 image through downsampled `out_shape`; it does not emit the 4096 × 4096 source. Statistics visit 256 blocks for the tested band and complete without a full-raster `src.read()` allocation. The test completed without an out-of-memory condition or abnormal process failure.

Uploads are streamed to disk in 1 MiB chunks and capped at 512 MiB, avoiding a second whole-file in-memory upload copy.

## 10. Agent Workflow

Real Qwen3-VL-4B decisions on the RTX 4090 passed these natural-language workflows:

- `检查这幅遥感影像的基本信息。` -> `inspect_raster`
- `生成这幅遥感影像的 RGB 预览。` -> `inspect_raster -> raster_preview`
- `检查这幅影像，并描述主要地物。` -> `inspect_raster -> raster_preview -> analyze_image`
- `裁剪影像左上四分之一区域，并统计各波段。` -> `inspect_raster -> crop_raster -> raster_statistics`

The visual-understanding case passed the preview artifact ID to `analyze_image`; it never passed GeoTIFF bytes to Qwen. YOLO11, YOLOE, and SAM load counts remained zero in raster-only validation.

The final independent Phase 7 process completed in 851.34 seconds: metadata 56.29 seconds, preview 166.74 seconds, preview visual analysis 298.88 seconds, and crop/statistics 325.57 seconds. Rasterio tool time was milliseconds; Qwen planning and final generation dominated elapsed time. The model loaded once and was reused across all four cases.

## 11. UI

The existing Gradio screen now accepts either an ordinary image or a `.tif`/`.tiff` upload. It shows Raster Information, generated Preview, Workflow steps, final answer, active raster/image IDs, and artifact parent links. Raster Information includes size, band count and dtypes, CRS, resolution, bounds, NoData, and available band statistics.

The manual tool panel exposes one-based bands, stretch percentiles, preview size, resampling, named raster regions, and explicit pixel-window coordinates.

Live Gradio validation passed with `inspect_raster -> raster_preview`, retained `raster-001` as `active_raster`, selected `raster-preview-001` as `active_image`, and verified both Agent and manual-tool panels. Live FastAPI validation passed `inspect_raster -> crop_raster -> raster_statistics`, including the 128 × 96 crop and preserved georeferencing.

## 12. Tests

Final ordinary validation:

```text
python -m compileall backend frontend scripts                         PASS
pytest -m "not integration"                                          130 passed, 13 deselected
pip check                                                             PASS
```

Each real GPU group ran in its own Python process:

```text
backend/tests/test_detection_integration.py                            2 passed in 477.22s
backend/tests/test_phase5_integration.py                               5 passed in 361.19s
backend/tests/test_phase6_integration.py                               1 passed in 663.03s
backend/tests/test_phase7_integration.py                               1 passed in 851.34s
backend/tests/test_phase7_cross_phase_smoke.py                         1 passed in 421.76s
python -m scripts.verify_phase7_api                                   PASS
python -m scripts.verify_phase7_ui                                    PASS
```

The cross-phase smoke used one real Qwen load. Smoke A completed `analyze_image -> final`; Smoke B completed `inspect_raster -> raster_preview -> analyze_image -> final`. Detector, open-vocabulary detector, and SAM load counts stayed zero.

## 13. GPU Integration Strategy

Heavy GPU integration is isolated by phase and file. Every integration file runs in a new pytest Python process so process exit releases its CUDA context, model singletons, cached tensors, and allocator state. FastAPI and Gradio remain stopped during pytest GPU integration; Live checks run only after integration processes finish.

A single independent GPU integration pytest process has a 30-minute hard limit. It is stopped earlier after 15 minutes without progress, unexplained sustained use above 22 GiB, a CUDA/OOM/device error, or visible Windows instability. A timeout is recorded as `TIMEOUT / PERFORMANCE ISSUE`, not an immediate functional failure. The timeout is not increased to hide slow generation. After a CUDA/device failure, the process is ended and any retry uses a new clean process after checking the current scenario, `nvidia-smi`, utilization, VRAM, Python processes, recent Agent/Tool metrics, repeated generation, and repeated model loading.

The former combined P0–P7 run lasted about 89 minutes, sustained 100% utilization, reached approximately 23.47 GiB physical VRAM, and later encountered `CUDA unknown error`. Failures after that damaged context were cascading environment failures. The error did not reproduce in the clean Phase 7 process, cross-phase smoke, FastAPI Live, or Gradio Live runs.

Observed clean-run physical VRAM during the final smoke and Live checks was about 23.4 GiB while Qwen was resident and generating, then returned to about 0.54 GiB after the processes exited. Response metadata can aggregate manager-level peak counters and therefore reported values above physical capacity in some multi-step runs; `nvidia-smi` was used for physical-device observations. Qwen `load_count` was 1 in the cross-phase smoke, and no unnecessary YOLO11/YOLOE/SAM load occurred. There is no evidence of duplicate model loading or a persistent P7 memory leak; the long combined process remains unsuitable as a CUDA health test.

## 14. Regression

The Phase 4 planner regression was fixed generically. The planner now checks every explicit user sub-goal before finalizing, so a request that combines detection and broader visual analysis continues from detection to `analyze_image` before `final`. The implementation does not use prompt-keyword routing or a fixed scenario branch. The isolated Phase 4 integration passed after the fix.

Ordinary `.jpg` and `.png` inputs retain the existing image tools and artifact semantics. The cross-phase smoke confirmed the ordinary-image path after all raster changes. No detector, open-vocabulary, segmentation, model-loading, or image-crop implementation was replaced.

## 15. Known Limitations

Phase 7 supports GeoTIFF raster foundations only. It does not map detections from preview pixel space back to source-raster pixels. It does not perform coordinate queries, reprojection, vector I/O, polygon clipping, raster algebra, spectral indices, real-area measurement, distance, buffers, spatial relations, spatial joins, or GIS database operations. Preview band selection is generic and does not infer sensor-specific band semantics.

Qwen planning and final generation remain the dominant performance cost. The isolated Phase 7 process is now below the 30-minute limit, but response latency is still high for interactive use.

## 16. Final Result

Phase 7 is PASS. The planner regression, ordinary P0–P7 tests, isolated Phase 4/5/6/7 GPU integrations, cross-phase GPU smoke, FastAPI Live, and Gradio Live all passed. The previous `CUDA unknown error` is classified as a long-running combined-suite CUDA context/environment failure, not a Phase 7 functional failure.