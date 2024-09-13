# ISIC-2024 Challenge (Kaggle)

This repository is a "shallow fork" of a private competition repository
containing my part of the work only on the team "fx-9860GII" for the ISIC 2024
Challenge hosted on
[Kaggle](https://www.kaggle.com/competitions/isic-2024-challenge/overview). The
aim of the competition is to predict whether a skin lesion is benign or
malignant, based on provided images and metadata of the lesion.

## Results

The final selected solutions were based on gradient-boosting methods. The lesion
images first passed into CV models to yield vision-only predictions. These
predictions are concatenated to the feature-engineered metadata table as input
for three different gradient boosting frameworks. Finally the predictions from
the three frameworks are combined together to give a final score.

- Stratified 4-fold cross-validation trained BEiT -> gradient boosting: 0.17353
  (public leaderboard (LB)), 0.16030 (private LB)
- ViT, ViTMAE, BEiT, EfficientNet-b7 -> gradient boosting: 0.17624 (public LB),
  0.15708 (private LB)

## Directories

- `docs`: Some notes.
- `preprocess_data`: Data preprocessing, basic dataset statistics and my
  attempts to combine data available on the ISIC Archive this year's new data.
- `scripts`: Scripts for remote GPU instance setup
- `src`: Most of the training and evaluation scripts.
