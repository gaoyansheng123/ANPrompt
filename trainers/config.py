def get_dataset_specified_config(dataset):
    """Get dataset specific."""
    cfg = {
        "StanfordCars": {
            "TRAINER.MMRL.BETA": 0.7,
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 7.0,
        },
        "FGVCAircraft": {
            "TRAINER.MMRL.BETA": 0.9,
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 6.0,
        },
        "SUN397": {
            "TRAINER.MMRL.BETA": 0.5,
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 6.0,
        },
        "DescribableTextures": {
            "TRAINER.MMRL.BETA": 0.9,
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 6.0,
           
        },
        "Food101": {
            "TRAINER.MMRL.BETA": 0.1,
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 5.0,
        },
        "OxfordFlowers": {
            "TRAINER.MMRL.BETA": 0.4,
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 4.0,
        },
        "UCF101": {
            "TRAINER.MMRL.BETA": 0.9,
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 3.0,
        },
        "ImageNet": {
            "TRAINER.MMRL.BETA": 0.9,
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 0.5,
        },
        "Caltech101": {
            "TRAINER.MMRL.BETA": 0.6,
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 0.5,
        },
        "OxfordPets": {
            "TRAINER.MMRL.BETA": 0.7,
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 0.2,
        },
        "EuroSAT": {
            "TRAINER.MMRL.BETA": 0.2,
            "TRAINER.MMRL.REP_DIM": 512,
            "TRAINER.MMRL.N_REP_TOKENS": 5,
            "TRAINER.MMRL.REG_WEIGHT": 0.001,
        },
    }.get(dataset, {})

    return [item for pair in cfg.items() for item in pair]