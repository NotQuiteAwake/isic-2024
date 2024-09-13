from transformers import ViTMAEForPreTraining, ViTForImageClassification
from base import get_model_init, get_training_args, get_trainer
from sets import prepare_full_resampled, _prepare_2024_resampled

# train, val = _prepare_2024_resampled(samp=(1, 1))
train, val = prepare_full_resampled(samp_cp=(2, 1), samp_24=(2, 5))
train_unsup = train.select_columns(["pixel_values"])
val_unsup = val.select_columns(["pixel_values"])

# unsupervised learning
model_init = get_model_init(
    ViTMAEForPreTraining,
    model_source="facebook/vit-mae-large",
    load_dir="model/vitmae-unsup/",
    eval_metric="eval_loss",
)
training_args = get_training_args(
    batch_size=32,
    save_dir="model/vitmae-unsup/",
    label_names=["pixel_values"],
    metric_for_best_model="loss",
)

trainer = get_trainer(
    model_init, training_args, train_unsup, val_unsup, compute_metrics=None
)
trainer.train()


# load the weights back as ViT classifier
model_init = get_model_init(
    ViTForImageClassification,
    model_source="facebook/vit-mae-large",
    load_dir="model/vitmae-unsup/",
    eval_metric="eval_loss",
)
training_args = get_training_args(batch_size=32, save_dir="model/vitmae-sup/")

trainer = get_trainer(model_init, training_args, train, val)
trainer.train()
