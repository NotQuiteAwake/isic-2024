from transformers import ViTForImageClassification
from base import get_model_init, get_training_args, get_trainer
from sets import prepare_full_resampled, prepare_2024_resampled

model_init = get_model_init(
    ViTForImageClassification,
    model_source="google/vit-base-patch16-224",
    load_dir="model/vit/",
)
training_args = get_training_args(
    batch_size=64, save_dir="model/vit-balanced/", learning_rate=2e-5
)

train, val = prepare_full_resampled(samp_cp=(1, 1), samp_24=(1, 10))

trainer = get_trainer(model_init, training_args, train, val)
trainer.train()


model_init = get_model_init(
    ViTForImageClassification,
    model_source="google/vit-base-patch16-224",
    load_dir="model/vit-balanced/",
)
training_args = get_training_args(
    batch_size=64, save_dir="model/vit-2024/", learning_rate=1e-5
)
train, val = prepare_2024_resampled(samp=(1, 10))
trainer = get_trainer(model_init, training_args, train, val)
trainer.train()
