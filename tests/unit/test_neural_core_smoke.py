import json

import torch

from hybrid_drive.experiments import neural_core_smoke as smoke


def test_real_cpu_optimizer_freezes_fixture_models_updates_core_and_roundtrips(tmp_path):
    root, result = smoke.run_neural_core_smoke(tmp_path, revision="a" * 40, steps=4, seed=0)
    assert result["status"] == "passed", result
    assert result["completed_steps"] == 4
    assert all(result["updated_groups"].values())
    assert result["frozen_parameters_unchanged"]
    assert result["frozen_parameters_with_gradients"] == 0
    assert result["roundtrip_bitwise_equal"]
    assert result["aligned_checkpoint_claim"] is result["driving_quality_claim"] is False
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["scope"] == smoke.SCOPE and manifest["real_road_data"] is False
    rows = json.loads((root / "steps.json").read_text())
    assert len(rows) == 4
    assert all(value > 0 for row in rows for value in row["gradient_norms"].values())
    assert (root / "core.safetensors").is_file() and (root / "report.html").is_file()


def test_synthetic_fixture_is_reproducible_and_does_not_mutate_caller_rng(tmp_path):
    torch.manual_seed(91)
    state = torch.random.get_rng_state().clone()
    _, first = smoke.run_neural_core_smoke(tmp_path, revision="a" * 40, steps=2, seed=1)
    assert torch.equal(torch.random.get_rng_state(), state)
    _, second = smoke.run_neural_core_smoke(tmp_path, revision="a" * 40, steps=2, seed=1)
    assert first["status"] == second["status"] == "passed", (first, second)
    assert first["output_sha256"] == second["output_sha256"]
    assert first["initial_loss"] == second["initial_loss"] and first["final_loss"] == second["final_loss"]


def test_accidental_no_grad_around_frozen_lm_is_detected(tmp_path, monkeypatch):
    original = smoke._FrozenFixtureLM.forward

    def detached(self, prefix, ids):
        hidden, logits = original(self, prefix, ids)
        return hidden.detach(), logits.detach()

    monkeypatch.setattr(smoke._FrozenFixtureLM, "forward", detached)
    root, result = smoke.run_neural_core_smoke(tmp_path, revision="a" * 40, steps=1, seed=0)
    assert result["status"] == "failed"
    assert "absent or nonfinite gradients" in result["error"]
    assert result["completed_steps"] == 0
    assert not (root / "core.safetensors").exists()
