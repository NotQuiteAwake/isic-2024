---
title: Kaggle Upload
author: Jimmy
date: 29/08/2024
colorlinks: true
---

This is an outline of how I uploaded and ran April's model on Kaggle.

## Creating notebook on Kaggle

Go to Kaggle, click "Create" on the left-hand side. Select "New Notebook". In
the notebook you should find a tab on the right. In the "Input" section click
"Add Input", and search for ISIC 2024. Once it's added, data will be available
at `/kaggle/input/isic-2024-challenge/`; The file of interest is
`test-image.hdf5` and the `test-metadata.csv` files.

## Model Download and Upload

The Kaggle scoring environment will be without internet, so we have to load an
offline copy of the model with `from_pretrained`, instead of from the repo. 

I used `huggingface-cli` to download it:

```{.bash}
huggingface-cli download April1234/ViT_base_batch_64 --local-dir vit_base_64
```

The `--local-dir` part is required, or else it will be downloaded to the HF
cache directory and give you a headache.

Now navigate to your Kaggle notebook again, on the right-hand tab click
"Upload", select "New Model". Select all files from the model directory you just
downloaded then upload. On the next page fill out the information; I was lazy so
I just picked "Other license" and used framework "Transformers". I would suggest
just using MIT License for the actual thing.

Your model will then also appear in `/kaggle/input/`, to see what the name of
the folder is add a line like this in your notebook then run (only just learnt
this is how you run shell commands in a jupyter notebook environment):

```{.bash}
!ls /kaggle/input/
```

You can also see what the directories should _approximately_ look like by
looking at the "Input" section on the right. I named the uploaded model
`ViT_base_batch_64`, and in this tab it is showing
`ViT_base_batch_64/default/v1`, and the actual directory is
`/kaggle/input/vit_base_batch_64/transformers/default/1/`.

## Write (or just copy) and run inference code

My code is available at `src/submit.ipynb` in the repository. My major reference
was [this
notebook](https://www.kaggle.com/code/nosherwantahir/notebookea3cca46ba), which
just popped up when I tried to upload the ISIC dataset myself (you don't need to
do this! Follow previous sections).

Some things to note, see
[this](https://www.kaggle.com/competitions/isic-2024-challenge/overview/evaluation)
and
[this](https://www.kaggle.com/competitions/isic-2024-challenge/overview/code-requirements):

- The test set will be in an `.hdf5` file for which `h5py` library is required.
- The photos are saved as bytes objects in the `.hdf5` files.
- Can't load anything from the internet as your notebook will be rerun in an
  offline environment when you submit. Also caching anything, or storing stuff
  in `/kaggle/working` will likely not work (not sure about the latter, but this
  is likely emptied every time you rerun a new version of the model)
- Submission file is called `submission.csv` with a specific format which you
  can obtain by running `pandas.DataFrame.to_csv('submission.csv', index = False)`.
  Column names are `isic_id` and `target`, the latter being the _malignant
  probability_.

## Submit your notebook for scoring

After you've done preliminary checks on your notebook, click "Save version" on
the top right. The saved version will be run in a new container, and you can
click on the number to the right of the "Save version" button to check the
all versions you've saved. When the version has finished running refresh the
page and you can see its output. If it looks sensible enough, click the "..."
right to your version, then "Submit to Competition". Track progress with "View
Active Events" on the bottom left. Find these notebooks later in "Your work" on
the left-hand bar.
