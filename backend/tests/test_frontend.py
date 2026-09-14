import inspect
import json

import httpx

from frontend.app import (
    PLACEHOLDER, _agent_markdown, _post_agent, _raster_markdown, analyze,
    build_manual_form_fields,
    _ui_safe_result, chat, fetch_status, manual_parameter_visibility,
    manual_tool_definitions,
)


def test_chat_placeholder():
    text, history = chat("hello", [])
    assert text == ""
    assert history == [{"role": "user", "content": "hello"}, {"role": "assistant", "content": PLACEHOLDER}]
    assert chat("  ", history)[1] == history


def test_backend_unavailable(settings, monkeypatch):
    def unavailable(*args, **kwargs):
        raise httpx.ConnectError("backend stopped")
    monkeypatch.setattr(httpx.Client, "get", unavailable)
    title, details = fetch_status(settings)
    assert "后端不可用" in title
    assert details == {"backend": "unavailable"}


def test_analysis_boundary_uses_tool_api_only():
    source = inspect.getsource(_post_agent) + inspect.getsource(analyze)
    assert "/agent/run" in source
    assert "/tools/" not in source
    assert "/models/vlm/infer" not in source


def test_agent_panel_shows_detection_and_segmentation_summary():
    panel = _agent_markdown({
        "run_id": "run-1",
        "steps": [{
            "index": 1, "decision_type": "tool_call", "tool_name": "detect_objects",
            "success": True, "duration_ms": 12.0,
            "observation_summary": {
                "detection_count": 5, "class_counts": {"bus": 1, "person": 4}
            },
        }, {
            "index": 2, "decision_type": "tool_call", "tool_name": "detect_open_vocab",
            "success": True, "duration_ms": 20.0,
            "observation_summary": {
                "detection_count": 1, "class_counts": {"yellow safety helmet": 1}
            },
        }, {
            "index": 3, "decision_type": "tool_call", "tool_name": "segment_objects",
            "success": True, "duration_ms": 30.0,
            "observation_summary": {
                "segment_count": 1, "segments": [{"mask_area_ratio": 0.125}],
                "failures": [{"code": "SEGMENTATION_INFERENCE_FAILED"}],
            },
        }],
        "metadata": {},
    })
    assert "检测总数：**5**" in panel
    assert "bus × 1" in panel and "person × 4" in panel
    assert "开放检测总数：**1**" in panel and "yellow safety helmet × 1" in panel
    assert "分割实例：**1**" in panel and "失败：**1**" in panel and "12.50%" in panel


def test_ui_result_redacts_machine_local_paths():
    safe = _ui_safe_result({
        "artifacts": [{
            "path": "E:/sht/DEMO/GeoAgent/outputs/run/mask.png",
            "mime_type": "image/png",
        }],
        "data": {
            "source_image_path": "D:/sht/DEMO/GeoAgent/test/img/1.jpg",
            "mask_artifact_path": "E:/sht/DEMO/GeoAgent/outputs/run/mask.png",
        },
    })
    assert safe["artifacts"][0]["artifact_path"] == "mask.png"
    assert safe["data"] == {
        "source_image_path": "1.jpg", "mask_artifact_path": "mask.png"
    }
    assert "D:/" not in json.dumps(safe) and "E:/" not in json.dumps(safe)


def test_manual_tool_fields_are_driven_by_registry_schema():
    definitions = manual_tool_definitions()
    assert set(definitions) == {
        "inspect_image", "crop_image", "analyze_image", "detect_objects",
        "detect_open_vocab", "segment_objects", "inspect_raster", "raster_preview",
        "crop_raster", "raster_statistics",
    }
    expected = {
        "inspect_image": set(),
        "crop_image": {"x1", "y1", "x2", "y2"},
        "analyze_image": {"prompt", "max_new_tokens"},
        "detect_objects": {"classes", "confidence", "iou_threshold"},
        "detect_open_vocab": {"classes", "confidence", "iou_threshold"},
        "segment_objects": {"boxes", "detection_ids"},
        "inspect_raster": set(),
        "raster_preview": {
            "bands", "stretch", "lower_percentile", "upper_percentile",
            "max_size", "resampling",
        },
        "crop_raster": {"region", "row_start", "row_end", "col_start", "col_end"},
        "raster_statistics": {"bands"},
    }
    for name, fields in expected.items():
        visible = manual_parameter_visibility(definitions[name])
        assert {key for key, shown in visible.items() if shown} == fields


def test_manual_form_serializes_only_selected_tool_schema():
    definition = manual_tool_definitions()["detect_open_vocab"]
    fields = build_manual_form_fields(definition, {
        "prompt": "ignored", "max_new_tokens": 512,
        "classes": "a person wearing a yellow helmet, excavator",
        "confidence": 0.25, "iou_threshold": 0.45,
        "x1": 1, "y1": 2, "x2": 3, "y2": 4, "boxes": "[]",
    })
    assert json.loads(fields["classes"]) == [
        "a person wearing a yellow helmet", "excavator"
    ]
    assert fields["confidence"] == 0.25 and fields["iou_threshold"] == 0.45
    assert not {"prompt", "max_new_tokens", "x1", "boxes"}.intersection(fields)


def test_raster_panels_show_metadata_statistics_and_artifact_flow():
    result = {
        "run_id": "raster-run",
        "steps": [{
            "index": 1, "decision_type": "tool_call", "tool_name": "inspect_raster",
            "success": True, "duration_ms": 2.0,
            "observation_summary": {"width": 256, "height": 192, "band_count": 3,
                                    "crs": "EPSG:32647", "source_artifact_id": "raster-001"},
        }, {
            "index": 2, "decision_type": "tool_call", "tool_name": "raster_statistics",
            "success": True, "duration_ms": 3.0,
            "observation_summary": {"bands": [{"band": 1}], "read_strategy": "block_windows",
                                    "source_artifact_id": "raster-001"},
        }],
        "workflow": {
            "active_image_artifact_id": "raster-preview-001",
            "active_raster_artifact_id": "raster-001",
            "raster": {
                "artifact_id": "raster-001", "artifact_type": "raster",
                "metadata": {"width": 256, "height": 192, "band_count": 3,
                             "dtypes": ["uint16"] * 3, "crs": "EPSG:32647",
                             "resolution_x": 10, "resolution_y": 10, "nodata": 0,
                             "bounds": {"left": 100, "bottom": 200, "right": 2660, "top": 2120}},
                "statistics": {"bands": [{"band": 1, "min": 3, "max": 15,
                                           "mean": 9, "std": 4, "valid_pixel_count": 13,
                                           "nodata_count": 3}]},
            },
            "artifacts": [], "categories": {},
        },
    }
    workflow_panel = _agent_markdown(result)
    raster_panel = _raster_markdown(result)
    assert "当前 Raster：`raster-001`" in workflow_panel
    assert "Raster：**256 × 192**" in workflow_panel
    assert "读取策略 `block_windows`" in workflow_panel
    assert all(text in raster_panel for text in (
        "Raster Information", "EPSG:32647", "256 × 192", "Band Statistics", "valid 13"
    ))
