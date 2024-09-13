from transformers import ResNetForImageClassification
from base import get_model_init, get_training_args, get_trainer
from sets import prepare_full_resampled

model_init = get_model_init(
    ResNetForImageClassification,
    model_source="microsoft/resnet-50",
    load_dir="model/resnet50/",
)
training_args = get_training_args(batch_size=128, save_dir="model/resnet50/")

train, val = prepare_full_resampled(samp_cp=(2, 2), samp_24=(2, 4))

trainer = get_trainer(model_init, training_args, train, val)
trainer.train()
