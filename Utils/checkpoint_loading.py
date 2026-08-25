def load_model_only(model, checkpoint_path, *, torch_module):
    """Load only model weights from a TS-DFM training checkpoint."""
    checkpoint = torch_module.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path!r} does not contain a 'model' state_dict"
        )

    model.load_state_dict(checkpoint["model"], strict=True)
    return {
        "checkpoint_path": str(checkpoint_path),
        "source_epoch": checkpoint.get("epoch"),
    }
