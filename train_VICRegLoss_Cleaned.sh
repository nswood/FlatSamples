#!/bin/bash


for ii in 7 10 15 20 50;
 do for jj in 0. 1 5 10;
    do 
      
      python3 IN_FlatSamples_VICRegLoss_Cleaned.py --weightstd 1 --weightrepr 1 --weightcov ${ii} --weightCorr1 ${jj} --weightCorr2 0 --nepochs 120
 done
done
