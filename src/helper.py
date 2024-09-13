import argparse
import subprocess

from transformers import AutoModelForImageClassification
from base import get_best_model_ckpt


def get_best_ckpt(args):
    model_dir, score = get_best_model_ckpt(args.checkpoint, args.eval_metric)
    print(f"model: {model_dir}")
    print(f"score: {score:.4f}")
    return model_dir, score


def get_commit_id():
    cmd = "git rev-parse --short HEAD".split()
    commit_id = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")
    return commit_id


def upload(args):
    model_dir, score = get_best_ckpt(args)
    model = AutoModelForImageClassification.from_pretrained(
        model_dir, num_labels=2, from_tf=False
    )
    commit_id = get_commit_id()

    print(f"model type: {type(model)}")
    print(f"commit: {commit_id}")

    model.push_to_hub(
        args.repo, private=True, commit_message=f"pauc {score:.4f}; commit {commit_id}"
    )


parser = argparse.ArgumentParser("Helper")
subparsers = parser.add_subparsers(help="Action you want to perform.")

parser_upload = subparsers.add_parser("upload", help="Upload model to HF.")
parser_upload.add_argument("checkpoint", help="Checkpoint folder.")
parser_upload.add_argument("repo", help="HF repository")
parser_upload.add_argument("eval_metric", nargs="?", default="eval_pauc")
parser_upload.set_defaults(func=upload)

parser_ckpt = subparsers.add_parser("checkpoint", help="Find best checkpoint.")
parser_ckpt.add_argument("checkpoint", help="Checkpoint folder.")
parser_ckpt.add_argument("eval_metric", nargs="?", default="eval_pauc")
parser_ckpt.set_defaults(func=get_best_ckpt)

args = parser.parse_args()
args.func(args)
