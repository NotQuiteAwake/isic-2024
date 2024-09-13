---
title: Notes on Data Processing
author: Jimmy
date: 23/08/2024
colorlinks: true
---

## Datasets

Currently there are three datasets available under the `TobanDjan` account on
HuggingFace. All image metadata processing and uploading is done in the script
`preprocess_data/combine.ipynb` on the `main` branch.

- The `isic_archive` and `isic_2024` sets are fully processed sets containing
  only `label` and `image` data fields, and without the other metadata.
- `isic_full` is the combined dataset of the previous two, with complete merged
  metadata from the two sets.

The `isic_archive` set is also referred to as the `complete` set, as it includes
everything from past challenges, HAM10000, and more.

All originally existing values are preserved.

The notebook as well as my other data analysis scripts relies on the helper
library `isic.py`, which contains the class `ISIC_Dataset` as well as other
useful methods.

### Additional tags

In addition to original metadata a few extra columns are added to store extra
information stemming from processing of data. All such columns are prefixed by
`PROC_` for processing.

- `PROC_source` contains the source of the data point. It can either be
  `complete` or `2024`.
- `PROC_use` contains intended use assigned to this data point. It can either be
  `training`, `validation` or `test`.

Other flags will be encountered below.

## `isic_archive`

This set is the full set of images that are usable for our training purposes
from the **ISIC Archive**. The input is fetched with `isic-cli` with the command
`isic image download`, which is equivalent to downloading all the images from
the ISIC Archive Gallery. For statistics and studies on dataset metadata see
`complete_set.ipynb`.

### `image_type` filter 

Non-`dermoscopic` images are removed. They have a very small population and we
cannot make sure they are of the same standards as dermoscopic ones.

### (`benign_malignant`, `diagnosis`) $\to$ `label`

Value in this column together with values in the `diagnosis` column jointly
determine the final `label`.

- `benign` $\to$ `label = 0`; `malignant` $\to$ `label = 1`.
- `indeterminate/malignant` discarded.
- `indeterminate`, `indeterminate/benign` $\to$ `label = 0`. This is supported by
  the mapping of all `indeterminate` samples to label 0 in the `2024` set.
- `nan` Guess benign/malignant from `diagnosis` column. Mapping based on
  benign/malignant statistics of the diagnosis in question in the `complete`
  set, cross-verified by `2024` statistics and GPT-queried benign/malignancy.
    - If diagnosis is also `nan` the row is removed as no sensible inference can
      be made of the data point. 
    - Rows with `label` inferred from diagnosis has `PROC_label_inferred` set to
      `1`.

For more details see `cp_label_maps` and `cp_nan_diag_maps` in `combine.ipynb`.

### Partitioning

$8:1:1$ train/validation/test set.

## `isic_2024`

This set is downloaded from the Kaggle competition page and is not available as
part of the ISIC Archive. The `zip` file is expanded with the script
`prepare_24.sh` to conform with the folder layout of other datasets setup with
the `ISIC_Dataset` class. Statistics and analyses are found in the script
`24_set.ipynb`.

### `target` $\to$ `label`

`target` column contains the inputs to our model, which in our protocol have the
name `label`.

### `iddx_1` $\to$ `benign_malignant`

`iddx_1` has values `Benign`, `Malignant` and `Indeterminate`. They are
downcased and mapped to `benign_malignant` to conform with the `isic_archive`
set.

### `iddx_3` $\to$ `diagnosis`

This column contains values similar to or overlapping with keys in the
`diagnosis` column of the diagnosis set, except that first letter is
upper-cased. GPT is fed both list of keys and prompted to draw connections from
`iddx_3` keys to `diagnosis` keys. These are verified by myself to (at least
appear to) make sense. Mappings which are not obvious are marked by a value 1 in
column `PROC_diag_inferred` A mapping is not _obvious_ when the mapped value is
not simply the downcased version of the original `iddx_3` value.

- Two or three keys of very small populations are mapped to `other`.
- `Solar or actinic keratosis` is mapped to `actinic keratosis`.
- Several subcategories are mapped to broader characterizations, eg `Melanoma in situ` $\to$ `Metanoma`

For detailed procedures check `isic.map_iddx3_diag`.

### Downsampling

All the malignant samples (`label` = 1) are kept. We sample 20x as many
benign/indeterminate images (`label` = 0).

### Partitioning

$4:1$ train/validation. No test set so that validation set has a reasonable
population of positives. The real test set is of course the final Kaggle test
set which is not available to us.

## `isic_full`

Produced by combining the fully-processed `isic_2024` and `isic_archive` sets.
Sets the `PROC_source` to `2024` or `complete` respectively so either set can be
retained or filtered out. Also uses the `PROC_use` value, set in the processing
of individual sets, to determine which subset (train/validation/test) to assign
a data point to.

### Non-existing columns

As the metadata tables are merged between `isic_2024` and `isic_archive`, their
columns are also merged together. Some columns are unique to `2024` while others
to `complete`. `pandas` assigns a null value (likely `np.nan`, but in any case
can be filtered with `df.isna()`) to previously non-existing values.

### dtype cleaning

In order for use by HF `Dataset`/`Arrow`, the `pandas` dataframe must have
uniform dtypes in each column. However, `np.nan` is of type `float64`.

- Columns with strings have their `nan` replaced by `''` (empty string).
- Columns with bools and `nan` have the bools cast to `float64`.
