def generate_transition1x_dataloaders(hdf5_file, batch_size, **loader_options):
    from Data.Transition1x import generate_dataloader_dynamics

    return generate_dataloader_dynamics(hdf5_file, batch_size, **loader_options)


def generate_mixed_dataloaders(**kwargs):
    from Data.MixedRGD1Transition1x import generate_mixed_dataloader_dynamics

    return generate_mixed_dataloader_dynamics(**kwargs)


def build_dynamics_dataloaders(config):
    data_config = config.data
    rgd1_cache_path = getattr(data_config, "rgd1_cache_path", None)
    if not rgd1_cache_path:
        return generate_transition1x_dataloaders(
            data_config.path,
            data_config.batch_size,
            num_workers=getattr(data_config, "num_workers", 0),
            pin_memory=getattr(data_config, "pin_memory", False),
            persistent_workers=getattr(data_config, "persistent_workers", False),
            prefetch_factor=getattr(data_config, "prefetch_factor", None),
        )
    return generate_mixed_dataloaders(
        transition1x_hdf5=data_config.transition1x_path,
        rgd1_cache_dir=rgd1_cache_path,
        batch_size=data_config.batch_size,
        seed=config.train.seed,
        num_workers=getattr(data_config, "num_workers", 0),
        pin_memory=getattr(data_config, "pin_memory", False),
        persistent_workers=getattr(data_config, "persistent_workers", False),
        prefetch_factor=getattr(data_config, "prefetch_factor", None),
    )
