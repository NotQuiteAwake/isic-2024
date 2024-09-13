from transformers import BeitForImageClassification
from base import get_model_init, get_training_args, get_trainer
from sets import prepare_full_kfold

for i, (train, val) in enumerate(prepare_full_kfold(samp_cp=(2, 1), samp_24=(2, 5),
                                               n_splits=4)):
    print(f"Fold {i}")
    model_dir = f"model/beit-{i}/"

    model_init = get_model_init(
            BeitForImageClassification,
            model_source="microsoft/beit-base-patch16-224-pt22k-ft22k",
            load_dir=model_dir
            )

    training_args = get_training_args(
        batch_size=64, save_dir=model_dir, learning_rate=3e-5
    )

    trainer = get_trainer(model_init, training_args, train, val)
    trainer.train()
