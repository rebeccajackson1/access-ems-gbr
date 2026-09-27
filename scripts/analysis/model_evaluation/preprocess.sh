#!/bin/bash
#PBS -l walltime=15:00:00,mem=100gb,ncpus=48
#PBS -P p66
#PBS -q normal
#PBS -m abe
#PBS -M rebecca.jackson@csiro.au
#PBS -j oe
#PBS -l wd
#PBS -lstorage=gdata/xp65+gdata/p66+gdata/et4+gdata/zv2

module use /g/data/xp65/public/modules
module load conda/analysis3-25.08

work_dir=/home/578/rj9627/python/shared/access-ems-gbr-evaluation

cd $work_dir
python3 -u preprocess.py > preprocess_status.log 2>&1

