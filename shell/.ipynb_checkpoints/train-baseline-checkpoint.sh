WORK_PATH="/home/amax/Codes/nmt-inter-state/THUMT"
MODEL_PATH="/home/amax/Data/nmt-inter-state/thumt/baseline_6"
mkdir -p $MODEL_PATH
DATA=$WORK_PATH/data/iwslt14_deen_10000
export PYTHONPATH=$WORK_PATH
device=1
src=de
tgt=en
time=$(date +'%m:%d:%H:%M')

# train
vinp=$DATA/valid.$src
vref=$MODEL_PATH/valid.$tgt.debpe
sed -r 's/(@@ )|(@@ ?$)//g' $DATA/valid.$tgt > $vref
python -u $WORK_PATH/thumt/bin/trainer.py \
  --input $DATA/train.$src $DATA/train.$tgt \
  --vocabulary $DATA/dict $DATA/dict \
  --model transformer \
  --validation $vinp \
  --references $vref \
  --output $MODEL_PATH \
  --parameters=device_list=[$device],train_steps=100000,batch_size=4096,log_steps=100,eval_steps=1000,update_cycle=1 \
  --half \
  --hparam_set iwslt14deen 2>&1 | tee $MODEL_PATH/train.log.$time


# inference
for file in $MODEL_PATH/eval/model-[0-9]*.pt
do
  model=${file##*/}
  inp=$DATA/test.$src
  ref=$MODEL_PATH/eval/test.$tgt.debpe
  hyp=$MODEL_PATH/eval/test.$src.out.$model
  sed -r 's/(@@ )|(@@ ?$)//g' $DATA/test.$tgt > $ref
  python -u $WORK_PATH/thumt/bin/translator.py \
    --models transformer \
    --input $inp \
    --output $hyp \
    --vocabulary $DATA/dict $DATA/dict \
    --checkpoints $MODEL_PATH/eval --specific-ckpt $MODEL_PATH/eval/$model \
    --parameters=device_list=[$device],decode_batch_size=32,decode_alpha=0.6,beam_size=5 2>&1 | tee $MODEL_PATH/eval/test.log.$model.$time
  sed -r 's/(@@ )|(@@ ?$)//g' $hyp > $hyp.debpe
  sacrebleu $ref -i $hyp.debpe -tok none -m bleu -w 4 -b > $MODEL_PATH/eval/bleu.log.$model.$time
done

