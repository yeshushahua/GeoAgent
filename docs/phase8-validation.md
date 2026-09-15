# Phase 8 Validation — Geospatial Intelligence

## 1. Goal

Phase 8 extends GeoAgent v0.9.0 from raster inspection into CPU geospatial analysis. The existing Agent can convert raster pixels to coordinates, create vector artifacts, calculate CRS-aware real area, and calculate raster statistics inside polygon zones.

No model, LangGraph, or multi-agent layer was added. Detection and segmentation implementations were not changed.

## 2. Architecture

The existing path remains:

```text
Natural language -> Qwen planner -> WorkflowController -> ToolRegistry
                 -> ToolExecutor -> CPU spatial tool -> Artifact
                 -> ResultAggregator -> final explanation
```

Four new tools use rasterio/GDAL/PROJ already present in the environment. They declare `requires_gpu=false` and `requires_model=null`.

## 3. Coordinate Awareness

`get_raster_coordinate` accepts an inspected raster artifact plus a zero-based `(row, col)`. It uses the pixel center, the dataset affine transform, and the source CRS. Its result contains:

- source CRS and EPSG
- projected/source `x` and `y`
- EPSG:4326 longitude and latitude
- the exact pixel and `center` offset convention

Tests cover EPSG:32647 UTM and EPSG:4326 geographic rasters, invalid pixels, and missing CRS.

## 4. GeoJSON Export

`export_geojson` accepts exactly one bbox, polygon, or binary mask. Pixel geometry is converted through the raster affine transform. A mask must match raster width and height exactly and is polygonized with rasterio.

The output is a GeoJSON FeatureCollection artifact with:

- geometry (`Polygon` or `MultiPolygon`)
- CRS name
- caller properties
- source type and coordinate-space provenance

The workflow registers it as `vector-001`, with type `vector`, role `vector_source`, and its source raster or mask as parent.

## 5. Real Area

`calculate_area` accepts a vector artifact or direct bbox/polygon/mask geometry. It does not use a fixed pixel-area assumption:

- projected CRS: planar polygon area multiplied by the CRS linear-unit conversion factor squared
- geographic CRS: geometry transformed to the local UTM zone before planar area calculation

It returns square metres, hectares, square kilometres, source CRS, calculation CRS, and calculation method. A known 100 m × 50 m polygon returns 5000 m² and 0.5 ha.

## 6. Zonal Statistics

`zonal_statistics` accepts a GeoTIFF plus a GeoJSON vector, bbox, or polygon. Vector geometry is transformed into raster CRS when needed. Every requested band is read by block window; a geometry mask selects raster-cell centers inside the zone.

Results contain count, mean, min, max, population standard deviation, band metadata, overlap status, CRS values, and read strategy. NoData, masked values, NaN, and Inf are excluded by raster masked reads and finite-value checks.

## 7. Spatial Artifacts

The workflow artifact types now include:

```text
raster -> vector -> analysis_result
   |                    ^
   +--------------------+
```

Coordinate, area, and zonal-statistics calls create `analysis-result-NNN` JSON artifacts. Workflow state exposes active vector/result IDs, metadata, parent provenance, completed actions, and compact observations. Public workflow views expose basenames and metadata without filesystem paths.

## 8. Workflow Integration

The Planner receives all four schemas dynamically. Its general rules require real CRS and grounded geometry, prefer `active_vector` after GeoJSON export, and prohibit conversion of segmentation pixel ratios into physical area.

The CPU scripted-planner acceptance path passed:

```text
inspect_raster
-> get_raster_coordinate
-> export_geojson
-> calculate_area
-> final
```

The final response includes structured spatial aggregation. A zonal-statistics workflow can use `active_raster` with `active_vector`. NDVI mean is supported only when the raster already contains an NDVI band; Phase 8 does not derive NDVI from generic bands.

## 9. Tool API

The generic Tool API accepts raster, GeoJSON, and mask uploads plus row/col, bbox, polygon, coordinate space, CRS, properties, bands, and `all_touched`. Uploaded temporary files are removed after execution; generated artifacts remain under configured output storage.

Structured errors include `CRS_REQUIRED`, `INVALID_PIXEL`, `INVALID_GEOMETRY`, `INVALID_VECTOR`, `CRS_TRANSFORM_FAILED`, `AREA_CALCULATION_FAILED`, and `ZONAL_STATISTICS_FAILED`.

## 10. CPU Test Results

Phase 8 focused tests:

```text
pytest backend/tests/test_phase8_geospatial.py -q
9 passed
```

Full ordinary regression before the version bump:

```text
pytest -m "not integration" -q
139 passed, 13 deselected
```

The suite covers UTM/geographic coordinate conversion, GeoJSON validity and CRS, mask polygonization, projected and geographic area, known zonal statistics, Tool discovery, API execution, Agent sequencing, Artifact IDs, roles, and provenance.

No GPU integration, model inference, FastAPI service process, or Gradio service process was started.

## 11. Compatibility

Phase 0–7 ordinary tests remain green. Existing image, detection, open-vocabulary, segmentation, raster, and Phase 7 GPU-testing behavior is unchanged. The Phase 0–7 Gradio manual debug forms remain limited to their existing inputs; Phase 8 tools are available through the natural-language Agent and generic Tool API.

## 12. Limitations

Phase 8 supports Polygon and MultiPolygon analysis. It does not implement point/line vector analytics, vector overlay, spatial joins, arbitrary reprojection exports, polygon clipping, raster algebra, or spectral-index generation.

Raster-aligned mask export requires exact source-raster dimensions. A mask produced from a downsampled preview is not silently mapped back to source pixels. Geographic area uses a local UTM projection and is intended for local/regional polygons, not antimeridian-spanning or near-global geometry. Zonal statistics use pixel-center inclusion by default; `all_touched=true` is available through the Tool API.

## 13. Result

Phase 8 is PASS. The final GeoAgent v0.9.0 ordinary suite completed with 139 passed and 13 GPU integrations deselected. Compile, dependency, and diff checks are green; no GPU work was executed.