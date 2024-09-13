import numpy as np
import pandas as pd
import os
from scipy.special import softmax

from sklearn.metrics import roc_curve, auc, roc_auc_score
import evaluate

from transformers import (
    PreTrainedModel,
    AutoImageProcessor,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
    DefaultDataCollator,
)
from transformers.trainer_callback import TrainerState


def score(solution: list, submission: list, min_tpr: float = 0.80) -> float:

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(np.asarray(solution) - 1)

    # flip the submissions to their compliments
    v_pred = -1.0 * np.asarray(submission)

    max_fpr = abs(1 - min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)

    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    return partial_auc


confusion_metric = evaluate.load("confusion_matrix")
accuracy_metric = evaluate.load("accuracy")


def compute_metrics(eval_prod):
    logits, labels = eval_prod
    probs = softmax(logits, axis=1)[:, 1]
    predictions = np.argmax(logits, axis=1)

    accuracy = accuracy_metric.compute(references=labels, predictions=predictions)

    confusion = confusion_metric.compute(
        references=labels, predictions=predictions, labels=[0, 1]
    )

    # or else have the JSON "can't be serialised" with np.array
    confusion["confusion_matrix"] = confusion["confusion_matrix"].tolist()

    pauc = {"pauc": score(solution=labels, submission=probs)}

    # must return single dict
    return confusion | accuracy | pauc


def get_best_model_ckpt(load_dir, eval_metric="eval_pauc") -> tuple[str | None, float]:
    if not os.path.exists(load_dir):
        return None, 0

    ckpt_dirs = os.listdir(load_dir)
    ckpt_dirs = [name for name in ckpt_dirs if "checkpoint" in name]
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))

    best_checkpoint = None
    best_metric = 0

    for cdir in ckpt_dirs:
        ckpt = f"{load_dir}/{cdir}/"
        state = TrainerState.load_from_json(f"{ckpt}/trainer_state.json")
        metric = state.log_history[-1][eval_metric]
        if metric > best_metric:
            best_metric = metric
            best_checkpoint = ckpt

    return best_checkpoint, best_metric


def get_model_init(
    model_class: type[PreTrainedModel], model_source, load_dir, eval_metric="eval_pauc"
):
    def model_init():
        best_model, _ = get_best_model_ckpt(load_dir=load_dir, eval_metric=eval_metric)
        load_model = model_source if best_model is None else best_model
        model = model_class.from_pretrained(
            load_model, num_labels=2, ignore_mismatched_sizes=True, from_tf=False
        )
        return model

    return model_init


def get_training_args(
    batch_size: int,
    save_dir: str,
    steps: int | None = None,
    seed: int = 42,
    nproc: int = 16,
    **kwargs,
):

    # for a batch size of 128 gives 128 steps per eval
    steps = steps if steps else int(2**14 / batch_size)
    # for batch size 128 gives 512 warmup steps
    warmup_steps = int(2**16 / batch_size)

    default_args = {
        "output_dir": save_dir,
        # lower these to 32 on local testing device
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "learning_rate": 5e-5,
        # also known as L2 regularization
        "weight_decay": 0.01,
        "max_steps": 100000,
        "warmup_steps": 640,
        "label_names": ["labels"],
        "metric_for_best_model": "pauc",
        "push_to_hub_model_id": None,
        "push_to_hub_organization": None,
        "eval_on_start": True,
        # must be the same for early stop
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "save_steps": steps,
        "logging_steps": steps,
        "eval_steps": steps,
        "save_total_limit": 3,
        "report_to": ["tensorboard"],
        "optim": "adamw_torch",
        # or else OOM
        "dataloader_persistent_workers": False,
        "dataloader_num_workers": nproc,
        "no_cuda": False,
        "use_cpu": False,
        "seed": nproc,
        "data_seed": nproc,
        # required for early stop.
        "load_best_model_at_end": True,
        "remove_unused_columns": False,
        "resume_from_checkpoint": True,
    }

    args = default_args | kwargs

    return TrainingArguments(**args)


def get_trainer(model_init, training_args, train, val, **kwargs):
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=10, early_stopping_threshold=1e-4
    )

    data_collator = DefaultDataCollator(return_tensors="pt")

    default_args = {
        "model_init": model_init,
        "args": training_args,
        "data_collator": data_collator,
        "train_dataset": train,
        "eval_dataset": val,
        "compute_metrics": compute_metrics,
        "callbacks": [early_stopping],
    }

    args = default_args | kwargs

    return Trainer(**args)
