from pemad_esb_inference.model_registry import ModelDefinition


def _definition(command):
    return ModelDefinition(
        key="k", region="us-central1", image="img", cpu=4, memory="16Gi",
        gpu=0, gpu_type=None, machine_type="c2-standard-4", timeout=360000,
        command=command, args=[],
    )


def test_family_is_viame_when_command_is_empty():
    assert _definition(command=[]).family == "viame"


def test_family_is_ultralytics_when_command_is_nonempty():
    assert _definition(command=["python", "infer.py"]).family == "ultralytics"


def test_family_heuristic_matches_dag_exactly_on_gpu_entries_too():
    # A GPU-enabled entry is still classified purely by `command`, not by
    # gpu/gpu_type -- confirms family and hardware are orthogonal, which is
    # the basis for saying ultralytics entries *could* be GPU-configured
    # without changing the family heuristic at all.
    gpu_definition = ModelDefinition(
        key="k", region="us-central1", image="img", cpu=4, memory="16Gi",
        gpu=1, gpu_type="nvidia-l4", machine_type="g2-standard-4", timeout=360000,
        command=[], args=["--model-name", "viame_detector"],
    )
    assert gpu_definition.family == "viame"
