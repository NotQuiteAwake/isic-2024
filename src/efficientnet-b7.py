from transformers import EfficientNetForImageClassification
from base import get_model_init, get_training_args, get_trainer
from sets import prepare_full_resampled, prepare_2024_resampled

model_init = get_model_init(
    EfficientNetForImageClassification,
    model_source="google/efficientnet-b7",
    load_dir="model/efficientnet-b7/",
)
training_args = get_training_args(
    batch_size=16, save_dir="model/en-balanced", learning_rate=2e-5
)

train, val = prepare_full_resampled(samp_cp=(1, 1), samp_24=(1, 10))
trainer = get_trainer(model_init, training_args, train, val)
trainer.train()

# further fine-tune on the 2024 dataset alone
model_init = get_model_init(
    EfficientNetForImageClassification,
    model_source="google/efficientnet-b7",
    load_dir="model/en-balanced/",
)
training_args = get_training_args(
    batch_size=16, save_dir="model/en-2024", learning_rate=1e-5
)
train, val = prepare_2024_resampled(samp=(1, 10))
trainer = get_trainer(model_init, training_args, train, val)
trainer.train()
