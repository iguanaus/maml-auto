#!/bin/bash
dirr=results_404040_002
note=" 
These are result breaking baseline. Note the innovation here was to make the meta_batch_size 2 which apparently drove the loss down to more manageable levels. 
python main.py --datasource=sinusoid --logdir=logs/sine7/  --norm=None --update_batch_size=100 --regularize_penal 0.0 --meta_lr .001 --update_lr .001 --metatrain_iterations=70000 --num_updates=20 --meta_batch_size=2 --limit_task=True --resume=True --train=False --test_set=True

"
mkdir $dirr
cd $dirr
echo "1.9E3779B97F4A7C15F39CC0" | pbcopy
echo "pass is on clip"
scp tor:/home/euler/John/maml-multi-combine/logs/sine7/cls_5.mbs_1.ubs_100.numstep20.updatelr0.001nonorm/train_loss.csv .
scp tor:/home/euler/John/maml-multi-combine/logs/sine7/cls_5.mbs_1.ubs_100.numstep20.updatelr0.001nonorm/validation_loss.csv .
scp tor:/home/euler/John/maml-multi-combine/logs/sine7/cls_5.mbs_1.ubs_100.numstep20.updatelr0.001nonorm/test_ubs100_stepsize0.001.csv .
echo "$note" > note.txt

