#!/bin/bash
dirr=results_404040_003
note=" 
These are the results for the training of the autoencoder. 

python main.py --datasource=sinusoid --logdir=logs/sine1/  --norm=None --update_batch_size=100 --regularize_penal 0.0 --meta_lr .001 --update_lr .001 --metatrain_iterations=0 --pretrain_iterations=70000 --num_updates=10 --meta_batch_size=25 --limit_task=True --resume=False

"
mkdir $dirr
cd $dirr
echo "1.9E3779B97F4A7C15F39CC0" | pbcopy
echo "pass is on clip"
scp tor:/home/euler/John/maml-auto/logs/sine1/cls_5.mbs_25.ubs_100.numstep10.updatelr0.001nonorm/train_loss.csv .
scp tor:/home/euler/John/maml-auto/logs/sine1/cls_5.mbs_25.ubs_100.numstep10.updatelr0.001nonorm/validation_loss.csv .
#scp tor:/home/euler/John/maml-auto/logs/sine1/cls_5.mbs_25.ubs_100.numstep10.updatelr0.001nonorm/test_ubs100_stepsize0.001.csv .
echo "$note" > note.txt

