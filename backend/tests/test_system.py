from backend.app.services import system


FIELDS = {"python_version", "torch_version", "cuda_available", "cuda_version",
          "gpu_name", "gpu_vram_gb", "project_root", "storage_root"}


def test_system(client):
    response = client.get("/api/v1/system")
    assert response.status_code == 200
    assert set(response.json()) == FIELDS


def test_cpu_only_returns_200(client, monkeypatch, caplog):
    monkeypatch.setattr(system.torch.cuda, "is_available", lambda: False)
    response = client.get("/api/v1/system")
    assert response.status_code == 200
    assert response.json()["cuda_available"] is False
    assert response.json()["gpu_name"] is None
    assert "CUDA unavailable" in caplog.text
