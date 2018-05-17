#!/bin/bash
dirr=results_404040_005/
note=" 
These are results from tor for trying to train it (prior to the big push). This got down to comparable leves to Tailin's model, but unfortunately is not very easily reconstructed. I will leave it here in case. 

"
mkdir $dirr
cd $dirr
#echo "1.9E3779B97F4A7C15F39CC0" | pbcopy
#echo "pass is on clip"

#scp tor:/home/euler/John/maml-auto/logs/sine1/cls_5.mbs_25.ubs_100.numstep10.updatelr0.001nonorm/validation_loss.csv .
#scp tor:/home/euler/John/maml-auto/logs/sine1/cls_5.mbs_25.ubs_100.numstep10.updatelr0.001nonorm/test_ubs100_stepsize0.001.csv .

#scp -r tor:/home/euler/John/maml-auto/logs/sine_backup1/cls_5.mbs_25.ubs_100.numstep10.updatelr0.001nonorm .

scp -r p2CalBig:/home/ubuntu/maml-auto/logs/sine9/cls_5.mbs_25.ubs_100.numstep5.updatelr0.001nonorm $dirr

echo "$note" > note.txt
