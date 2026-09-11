"""Exercise every Phase 1 model endpoint over real HTTP."""
import json

import httpx

from backend.app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    image = settings.project_root / "sample_data" / "images" / "phase1-spatial.png"
    records = {}
    with httpx.Client(timeout=600, trust_env=False) as client:
        records["initial_status"] = client.get(
            f"{settings.api_base_url}/models/vlm/status"
        ).json()
        load = client.post(f"{settings.api_base_url}/models/vlm/load")
        load.raise_for_status()
        records["load"] = load.json()
        with image.open("rb") as stream:
            infer = client.post(
                f"{settings.api_base_url}/models/vlm/infer",
                files={"image": (image.name, stream, "image/png")},
                data={
                    "prompt": "What is located in the upper-left area of the image?",
                    "max_new_tokens": 64,
                },
            )
        infer.raise_for_status()
        records["infer"] = infer.json()
        status = client.get(f"{settings.api_base_url}/models/vlm/status")
        status.raise_for_status()
        records["ready_status"] = status.json()
        unload = client.post(f"{settings.api_base_url}/models/vlm/unload")
        unload.raise_for_status()
        records["unload"] = unload.json()
    assert records["initial_status"]["state"] == "UNLOADED"
    assert records["load"]["state"] == "READY"
    assert records["infer"]["success"] and records["infer"]["text"]
    assert records["infer"]["device"] == "cuda:0"
    assert records["unload"]["state"] == "UNLOADED"
    target = settings.output_dir / "benchmarks" / "phase1" / "api_acceptance.json"
    target.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"API acceptance written to {target}")


if __name__ == "__main__":
    main()
