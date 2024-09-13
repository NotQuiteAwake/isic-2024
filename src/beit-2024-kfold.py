from transformers import BeitForImageClassification
from base import get_model_init, get_training_args, get_trainer
from sets import prepare_2024_kfold

for i, (train, val) in enumerate(prepare_2024_kfold(n_splits=4, samp=(10, 1))):
    print(f"Fold {i}")
    model_dir = f"model/beit-24-{i}/"

    model_init = get_model_init(
            BeitForImageClassification,
            model_source="microsoft/beit-base-patch16-224-pt22k-ft22k",
            load_dir=model_dir
            )

    # with about 4k samples 60 steps is about 1 epoch
    training_args = get_training_args(
        batch_size=64, save_dir=model_dir, learning_rate=2e-5, steps=64
    )

    trainer = get_trainer(model_init, training_args, train, val)
    trainer.train()
