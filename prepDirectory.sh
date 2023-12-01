#!/bin/bash

echo "###############################"
echo "##   Preparing Datasets      ##"
echo "###############################"
echo ""
wget --no-check-certificate https://huggingface.co/datasets/shashikg/visual_search_klab/resolve/main/dataset_asymmetry.zip -O ./dataset.zip
echo "Extracting..."
unzip dataset.zip -d ./vs_exp/

echo ""
echo "###############################"
echo "## Downloading VGG16 Weights ##"
echo "###############################"
echo ""
wget --no-check-certificate https://dl.dropboxusercontent.com/s/6r196ijkubpxchs/pretrained_model.zip?dl=1 -O ./pretrained_model.zip
unzip pretrained_model.zip

echo ""
echo "###############################"
echo "## Downloading GBVS Model    ##"
echo "###############################"
echo ""
wget --no-check-certificate https://dl.dropboxusercontent.com/s/v9tbt3b061iab6x/gbvs.zip?dl=1 -O ./gbvs.zip
echo "Extracting..."
unzip gbvs.zip -d ./vs_exp/

echo ""
echo "###############################"
echo "## Cleaning Directory...     ##"
echo "###############################"
echo ""
rm dataset.zip pretrained_model.zip gbvs.zip

echo ""
echo "###############################"
echo "## Prepare Conda Environment ##"
echo "###############################"
echo ""
conda env create -f environment.yml
