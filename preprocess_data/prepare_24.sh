#!/bin/bash

set -e

ARCHIVE="isic-2024-challenge.zip"
TEMP="2024-temp/"
NEW="2024/"

if [ -d "$TEMP" ] || [ -d "$NEW" ]; then
    echo "Directories exist, refusing to overwrite"
    exit 1
fi

unzip $ARCHIVE -d $TEMP

echo "Preparing new directory $NEW"

mkdir $NEW
mv $TEMP/train-image/image/ $NEW/training/
cp $TEMP/train-metadata.csv $NEW/training/metadata.csv
