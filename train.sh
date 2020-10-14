#! /bin/bash 

cd /tf/work/martin/Dropbox/projects/HowManyImages/
#rm logs.*
python train.py 2>&1 | tee -a logs.txt
