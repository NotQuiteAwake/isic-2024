import subprocess
import os

import pandas as pd
from pandas.api.types import is_object_dtype, is_numeric_dtype, is_bool_dtype

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEED = 6


def printdash(n_dashes: int = 20):
    dashes = n_dashes * "-"
    print(dashes)


def printenc(*args, **kwargs):
    printdash()
    print(*args, **kwargs, sep="\n")
    printdash()


def println(*args, **kwargs):
    """! @brief Just a wrapper to print an extra line lol."""
    print(*args, **kwargs)
    print()


def dtype_based_stats(ds: pd.DataFrame, col: str) -> str:
    stats = ""
    if is_numeric_dtype(ds[col]):
        stats = stats + str(ds[col].describe())
        if len(ds[col].unique()) <= 5:
            stats = stats + "\n" + str(ds[col].value_counts(dropna=False))
        else:
            stats = stats + "\nNumber of nan: " + str(ds[col].isna().sum())
            stats = stats + "\nUnique values: " + str(len(ds[col].unique()))
    else:
        stats = stats + str(ds[col].value_counts(dropna=False))

    return stats


class ISIC_Dataset:
    #     index_list: list[str] = ['isic_id', 'image_type', 'benign_malignant',
    #                              'PROC_use',
    #                              'PROC_source']

    def __init__(self, iden: str, cmd: list, use: str = "", root="data/isic/"):
        """!
        @brief initilialise a ISIC_Dataset object

        @param iden identifier for set, year / HAM10000 / complete
        @param cmd command for download, split for use by subprocess.Popen
        @param use original annotated use, test / training / validation
        @param root root of download directory
        """
        self.iden = iden
        self.cmd = cmd
        self.path = os.path.join(root, iden, use)

        USES = ["test", "training", "validation"]

        self.use = ""

        for trial_use in USES:
            if trial_use in use.lower():
                self.use = trial_use

    def __repr__(self) -> str:
        """! @brief defines format for printing."""
        return f"{vars(self)}"

    def download(self) -> subprocess.Popen:
        """! @brief download image and metadata in this set with isic command"""
        cmd = self.cmd + [self.path]
        return subprocess.Popen(cmd)

    """NOTE: methods below require download of the set to have completed"""

    def get_filename(self, isic_id: str):
        filename = os.path.join(self.path, isic_id + ".jpg")
        return filename

    def find_missing(self) -> list[str]:
        metadata = self.get_metadata()
        missing_ids = []
        for isic_id in tqdm(metadata["isic_id"]):
            filename = self.get_filename(isic_id)
            if not os.path.exists(filename):
                missing_ids.append(isic_id)

        return missing_ids

    def verify_integrity(self) -> bool:
        return len(self.find_missing()) == 0

    def get_metadata(self) -> pd.DataFrame:
        """! @brief Get raw metadata for this dataset"""

        # for eliminating mixed type warnings in pandas import
        DTYPES = {
            "concomitant_biopsy": "float",
            "fitzpatrick_skin_type": "string",
            "mel_class": "string",
            "mel_mitotic_index": "string",
            "mel_type": "string",
            "iddx_5": "string",
        }

        return pd.read_csv(os.path.join(self.path, "metadata.csv"), dtype=DTYPES)

    def get_image(self, isic_id: str, verify: bool = True) -> Image:
        """!
        @brief Get a single Image by isic_id (slow)

        @param isic_id isic_id string
        @param verify check image belongs in set before attempting access

        @return PIL image object

        @throws Exception raised if image is not in this dataset.

        This will be SLOW! Only use for previews. There's a reason this isn't
        called __getitem__(). If you want to actually access the images use
        get_images below instead.
        """

        if verify:
            if isic_id not in self.get_metadata()["isic_id"].values:
                raise Exception("Image not in this dataset!")

        filename = self.get_filename(isic_id)
        image = Image.open(filename)

        return image

    def get_images(self) -> list[dict[str, any]]:
        """!
        @brief return images in the dataset and their corresponding isic_id's

        @return list of images with corresponding isic_ids
        """
        metadata = self.get_metadata()
        images = []
        for isic_id in tqdm(metadata["fg"]):
            # verify is rather costly, we know these elements exist so don't
            # need this. If file is not found it's a much more serious issue
            # (actual stored data corrupted)
            image = self.get_image(isic_id, verify=False)
            images.append({"isic_id": isic_id, "image": image})

        return images

    def get_proc_metadata(self) -> pd.DataFrame:
        """!
        @brief get a processed metadata frame, with additional information

        @return table of metadata containing additional tags.
        """
        metadata = self.get_metadata()
        metadata["PROC_use"] = self.use
        metadata["PROC_source"] = self.iden
        # metadata = metadata[self.index_list]

        return metadata

    def get_image_dataframe(self) -> pd.DataFrame:
        """!
        @brief get dataframe with processed metadata and PIL image objects

        @return final dataframe containing images.
        """
        metadata = self.get_proc_metadata()
        images = self.get_images()
        combined_df = pd.merge(metadata, pd.DataFrame(images), on="isic_id")
        return combined_df


# COLLECTION_CMD = "isic collection list | sed '1,3d;$d;s/│/ /g' | grep Challenge"
# might as well do it more pythonically.
COLLECTION_CMD = "isic collection list"
SEP = "│"
CL_DOWNLOAD_CMD = "isic image download --collections"
FULL_DOWNLOAD_CMD = ["isic", "image", "download", "--search"]


def get_collections() -> list[ISIC_Dataset]:
    """!
    @brief Get all relevant ISIC "collections" as dataset objects.

    @return A list of dataset objects containing information of each collection.

    Collection is a ISIC archive concept. Relevant collections determined by the
    KEYWD variable below.
    """
    result = subprocess.run(
        COLLECTION_CMD.split(), capture_output=True, text=True, shell=False
    )

    GARBAGE = ["True", "False", "None"]
    KEYWD = ["Challenge", "HAM10000"]

    result = result.stdout.splitlines()
    collections = []

    for line in result:
        for garbage in GARBAGE:
            line = line.replace(garbage, "")

        # only keep lines that has keywords of interest
        for keywd in KEYWD:
            if keywd in line:
                line = line[1:].split(sep=SEP)
                line = [word.strip() for word in line if word.strip()]

                line[0] = int(line[0])  # collection id
                line.extend(line.pop().rsplit(" ", 1))
                line[1] = (
                    line[1]
                    .replace("Challenge ", "")
                    .replace("Task ", "_")
                    .replace(":", "")
                    .replace(" ", "")
                )

                use = "" if len(line) == 2 else line[2]
                col = ISIC_Dataset(
                    iden=line[1], cmd=CL_DOWNLOAD_CMD.split() + [str(line[0])], use=use
                )
                collections.append(col)
                break

    return collections


def get_complete(indeterminate: bool = True, dermoscopic: bool = False) -> ISIC_Dataset:
    """!
    @brief Get the full ISIC archive that is dermoscopic

    @return Dataset object containing information for the complete set.

    On second thought, since it is only the difference between applying the
    filter on their remote machine and applying on my own, it seems always
    superior to just download the whole set and filter with pandas.
    """

    DET_ARG = '(benign_malignant:"benign" OR benign_malignant:"malignant")'
    DERMO_ARG = 'image_type:"dermoscopic"'

    filter_args: list[str] = []
    if not indeterminate:
        filter_args.append(DET_ARG)
    if dermoscopic:
        filter_args.append(DERMO_ARG)

    filt = " AND ".join(filter_args)
    cmd = FULL_DOWNLOAD_CMD + [filt]
    complete_set = ISIC_Dataset(iden="complete", cmd=cmd)

    return complete_set


def get_2024() -> ISIC_Dataset:
    """!
    @brief Get the 2024 dataset object.

    @return an ISIC_Dataset carrying information to the 2024 dataset.

    The dataset can  be downloaded from Kaggle as an archive. Place that in
    data/isic/ and execute prepare_24.sh and the script will prepare dataset for
    you for use with this code.
    """

    ds = ISIC_Dataset(iden="2024", cmd=[], use="training")

    return ds


def get_all_datasets() -> list[ISIC_Dataset]:
    """!
    @brief Get all relevant datasets: collections and the complete set.

    @return all relevant datasets in a list.
    """
    return get_collections() + [get_complete()]


def download_sets(datasets: list[ISIC_Dataset]) -> list[int]:
    """!
    @brief download a number of datasets

    @param datasets datasets to download

    @return list of return values.

    If there are non-zeros in the return value then one of your downloads have
    probably failed. However I do not know if isic-cli properly set exit codes,
    so best to just check the final outputs from each download thread.
    """

    print(*datasets, sep="\n")

    subproc = []

    for ds in datasets:
        subproc.append(ds.download())

    exit_codes = [p.wait() for p in subproc]
    return exit_codes


def describe_dataset(ds: ISIC_Dataset):
    """!
    @brief Print some most basic information about a dataset.

    @param ds dataset of interest.
    """
    print(ds.iden)
    metadata = ds.get_metadata()
    print(list(metadata))
    print(metadata["benign_malignant"].describe())
    print()


def concat_metadata(datasets: list[ISIC_Dataset]) -> pd.DataFrame:
    """!
    @brief concatenate metadata of supplied dataset into one table.

    @param datasets a list of dataset objects

    @return a combined table.
    """
    # metadata = pd.DataFrame({index:[] for index in ISIC_Dataset.index_list})
    assert len(datasets)
    metadata = datasets[0].get_proc_metadata()
    for ds in datasets[1:]:
        metadata = pd.concat([metadata, ds.get_proc_metadata()])

    return metadata


def test_duplicates():
    """! @brief Investigate duplicates in datasets."""
    datasets = get_all_datasets()
    metadata = concat_metadata(datasets)
    print(metadata)

    dup = metadata[metadata.duplicated("isic_id", keep=False)]
    grouped = dup.groupby(["isic_id"])
    for name, group in grouped:
        print(name)
        print(group)
        print()


def test_complete():
    """!
    @brief Investigate completeness of the sets

    To be specific we want to compare if the union of all the other datasets is
    strictly inside the ISIC archive set.
    """
    complete = get_complete()
    comp_md = complete.get_proc_metadata()

    collections = get_collections()
    coll_md = concat_metadata(collections)

    #    coll_filt = coll_md[coll_md['benign_malignant'].isin(['benign', 'malignant'])]

    ids_comp = comp_md["isic_id"].unique()
    ids_coll = coll_md["isic_id"].unique()

    ids_comp_only = set(ids_comp) - set(ids_coll)

    printenc("Data unique to the full isic archive")
    comp_unique = comp_md[comp_md["isic_id"].isin(ids_comp_only)]
    println(comp_unique[["isic_id", "benign_malignant", "image_type"]])

    printenc("Data unique to the ISIC challenges + HAM10000", "")
    ids_coll_only = set(ids_coll) - set(ids_comp)
    coll_unique = coll_md[coll_md["isic_id"].isin(ids_coll_only)]
    println(coll_unique[["isic_id", "PROC_source", "benign_malignant"]])


#    print()
#    print(coll_md_unique['PROC_source'].value_counts(dropna=False))
#    print()
#    print(coll_md_unique['benign_malignant'].value_counts(dropna=False))


def preview_classes(
    ds: ISIC_Dataset,
    metadata: pd.DataFrame,
    group_name: str,
    select_cols: list[str] = ["isic_id", "benign_malignant", "image_type"],
):

    SAMPLE_SIZE = 4

    samples = metadata.groupby(group_name)
    for name, group in samples:
        print(name)
        group = group.sample(n=SAMPLE_SIZE, random_state=SEED)
        println(group[select_cols])

        fig, axs = plt.subplots(2, 2)

        for i in range(SAMPLE_SIZE):
            isic_id = group["isic_id"].iloc[i]
            axs[i // 2, i % 2].imshow(ds.get_image(isic_id))
            axs[i // 2, i % 2].set_title(isic_id)

        fig.suptitle(name)
        fig.tight_layout()
        plt.show()


def map_iddx3_diag(iddx_3_key: str, query_inferred: bool = False) -> str:
    # my weak mind would never have the strength to face this amount of work,
    # but GPT gladly does it in 10 seconds after I feed it the "unique" lists. AI! ML! LLMs!
    # The knowledge, the wisdom! Humanity's best friend, a flame in the darkest night!

    # Prompt is for GPT to make associations between unique keys in iddx_3 and
    # diagnosis. With the shared keys included in the "diagnosis keys" set the
    # same result can be obtained.
    iddx_3_map = {
        "angiofibroma": "angiofibroma or fibrous papule",  # Could also be 'angioma'
        "atypical intraepithelial melanocytic proliferation": "AIMP",  # Also 'atypical melanocytic proliferation'
        "atypical melanocytic neoplasm": "atypical melanocytic proliferation",  # Could also be 'AIMP'
        "fibroepithelial polyp": "acrochordon",
        "hemangioma": "angioma",  # General category
        "hidradenoma": None,  # No direct match in the diagnosis list
        "lichen planus like keratosis": "lichenoid keratosis",
        "melanoma Invasive": "melanoma",  # Also could be 'melanoma, NOS' or 'melanoma in situ'
        "melanoma in situ": "melanoma",  # Also could be 'melanoma Invasive' or 'melanoma, NOS'
        "melanoma, NOS": "melanoma",  # Also could be 'melanoma Invasive' or 'melanoma in situ'
        "solar or actinic keratosis": "actinic keratosis",  # TODO: THIS IS DEBATABLE
        "squamous cell carcinoma in situ": "squamous cell carcinoma",  # Also 'squamous cell carcinoma, Invasive' or 'squamous cell carcinoma, NOS'
        "squamous cell carcinoma, Invasive": "squamous cell carcinoma",  # Also 'squamous cell carcinoma in situ' or 'squamous cell carcinoma, NOS'
        "squamous cell carcinoma, NOS": "squamous cell carcinoma",  # Also 'squamous cell carcinoma in situ' or 'squamous cell carcinoma, Invasive'
        "trichilemmal or isthmic-catagen or pilar cyst": None,  # No direct match in the diagnosis list
    }

    inferred = -1
    diag = ""  # good flag for 'unprocessed' because '' is not a key in dataset

    # null diagnosis, we can do nothing
    if type(iddx_3_key) is not str and np.isnan(iddx_3_key):
        inferred = 0
        diag = iddx_3_key

    else:
        # formatting: the 2024 dataset has capitalised first letter while the rest
        # of the isic never does this
        lower_first_let = lambda s: s[0].lower() + s[1:]
        iddx_3_key = lower_first_let(iddx_3_key)

        # by processing this would only be if the original label is already in
        # archive sets
        if iddx_3_key not in iddx_3_map:
            inferred = 0
            diag = iddx_3_key

        # this label is not originally in the archive sets but its meaning is
        # equivalent or very close in meaning to another label in isic complete set
        elif iddx_3_map[iddx_3_key] is not None:
            inferred = 1
            diag = iddx_3_map[iddx_3_key]

        # mapping to None: just use other, these are small in population size
        # TODO: WE SHOULD DISCUSS THIS CHOICE. PERHAPS THIS SHOULD JUST GIVE NAN
        # Since other can be anything and just like nan it doesn't really provide us
        # information to make any inference/deduction of benign/malignancy.
        else:
            inferred = 1
            diag = "other"

    assert inferred != -1
    assert diag != ""

    if query_inferred:
        return inferred

    return diag


if __name__ == "__main__":
    ds24 = get_2024()
    metadata = ds24.get_metadata()
    columns = set(metadata.columns.values)

    complete = get_complete()
    c_meta = complete.get_metadata()
    c_columns = set(c_meta.columns.values)

    print(c_columns - columns)
    print(columns - c_columns)
    print(columns & c_columns)

# print(ds24)
# print(ds24.find_missing())
# print(metadata.columns.values)
