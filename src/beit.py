from transformers import BeitForImageClassification
from base import get_model_init, get_training_args, get_trainer
from sets import prepare_full_resampled

model_init = get_model_init(
    BeitForImageClassification,
    model_source="microsoft/beit-base-patch16-224-pt22k-ft22k",
    load_dir="model/beit/",
)
training_args = get_training_args(
    batch_size=64, save_dir="model/beit/", learning_rate=5e-6
)

train, val = prepare_full_resampled(samp_cp=(2, 1), samp_24=(2, 5))

trainer = get_trainer(model_init, training_args, train, val)
trainer.train()