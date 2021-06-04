#!/bin/bash

cd vs_exp/eccNET
echo ""
echo "#########################################"
echo "##   Running Exp using DL models       ##"
echo "#########################################"
echo ""
python runExp_ablation.py

cd ../pixelMatch
echo ""
echo "#########################################"
echo "##   Running Exp using pixelMatch      ##"
echo "#########################################"
echo ""
python runExp.py

cd ../chance
echo ""
echo "#########################################"
echo "##   Running Exp on chance prediction  ##"
echo "#########################################"
echo ""
python runExp.py
