class CFG:
    # General
    seed = 42
    batch_size = 4
    num_workers = 4  # For DataLoader parallelism

    # Audio Settings
    sample_rate = 16000  # 16kHz sampling rate
    duration = 10  # Target duration in seconds
    audio_len = sample_rate * duration  # Target length in samples (10s * 16kHz)

    # AST Model
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"  # AST pretrained model
    feature_extractor_name = model_name  # Feature extractor name (usually same as model)

    # Training
    lr = 1e-6  # Learning rate
    num_epochs = 30
    model_save_dir = "./saved_models"