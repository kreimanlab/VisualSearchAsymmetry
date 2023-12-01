#!/bin/bash

echo "###############################"
echo "##   Preparing Datasets      ##"
echo "###############################"
echo ""
wget --no-check-certificate https://huggingface.co/datasets/shashikg/visual_search_klab/resolve/main/dataset_saccade.zip -O ./dataset_saccade.zip
echo "Extracting..."
unzip dataset_saccade.zip -d ./vs_saccade_exp/

echo ""
echo "##########################################"
echo "## Downloading Additional Models        ##"
echo "##########################################"
echo ""
wget --no-check-certificate https://dl.dropboxusercontent.com/s/k0tv7jzxjxt1bzz/data_augmented_trained_model.zip?dl=1 -O ./data_augmented_trained_model.zip
echo "Extracting..."
unzip data_augmented_trained_model.zip

echo ""
echo "###############################"
echo "## Cleaning Directory...     ##"
echo "###############################"
echo ""
rm dataset_saccade.zip data_augmented_trained_model.zip

echo ""
echo "###############################"
echo "## Installing ScanMatchPy    ##"
echo "###############################"
echo ""
echo "[INFO]: You may need to provide sudo rights to install matlab runtime included in ScanMatchPy"
# eval "$(conda shell.bash hook)"
# conda activate vsa_nips_klab
# pip install git+https://github.com/kreimanlab/ScanMatchPy
