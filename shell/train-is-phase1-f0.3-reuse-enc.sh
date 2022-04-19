WORK_PATH="/home/amax/Codes/nmt-inter-state/THUMT"
MODEL_PATH="/home/amax/Data/nmt-inter-state/thumt/is_phase1.1_fratio0.3_reuse_enc"
mkdir -p $MODEL_PATH
DATA=$WORK_PATH/data/iwslt14_deen_10000_is
export PYTHONPATH=$WORK_PATH
device=1
src=de
tgt=en
time=$(date +'%m:%d:%H:%M')

# train
ratio=0.35
flip_ratio=0.3
vinp=$DATA/valid.$src
vref=$MODEL_PATH/valid.$tgt.debpe
sed -r 's/(@@ )|(@@ ?$)//g'  $DATA/valid.$tgt > $vref
python -u $WORK_PATH/thumt/bin/trainer.py \
  --input $DATA/train.$src $DATA/train.$tgt \
  --vocabulary $DATA/dict $DATA/dict \
  --model phase1transformer \
  --validation $vinp \
  --references $vref \
  --output $MODEL_PATH \
  --parameters=device_list=[$device],train_steps=200000,batch_size=4096,log_steps=100,eval_steps=1000,keep_top_checkpoint_max=5,update_cycle=1,normalization="after",phase=1,ratio=$ratio,flip_ratio=$flip_ratio,sample_train_mode=0,sample_infer_mode=0,reuse_encoder=True  \
  --advice $DATA/train.advices $DATA/valid.advices \
  --half \
  --hparam_set iwslt14deen 2>&1 | tee $MODEL_PATH/train.log.$time


# inference
for file in $MODEL_PATH/eval/model-[0-9]*.pt
do
  model=${file##*/}
  inp=$DATA/test.$src
  ref=$MODEL_PATH/eval/test.$tgt.debpe
  hyp=$MODEL_PATH/eval/test.$src.out.$model
  sed -r 's/(@@ )|(@@ ?$)//g'  $DATA/test.$tgt > $ref
  python -u $WORK_PATH/thumt/bin/translator.py \
    --models phase1transformer \
    --input $inp \
    --output $hyp \
    --vocabulary $DATA/dict $DATA/dict \
    --checkpoints $MODEL_PATH/eval --specific-ckpt $MODEL_PATH/eval/$model \
    --parameters=device_list=[$device],decode_batch_size=32,decode_alpha=0.6,beam_size=5,phase=1,ratio=$ratio,flip_ratio=$flip_ratio,sample_infer_mode=0,reuse_encoder=True \
    --advice $DATA/test.advices 2>&1 | tee $MODEL_PATH/eval/test.log.$model.$time

  sed -r 's/(@@ )|(@@ ?$)//g'  $hyp > $hyp.debpe
  sacrebleu $ref -i $hyp.debpe -tok none -m bleu -w 4 -b > $MODEL_PATH/eval/bleu.log.$model.$time

  inp=$DATA/test.$src
  ref=$MODEL_PATH/eval/test.$tgt.debpe
  hyp=$MODEL_PATH/eval/test.$src.out.$model.noad
  sed -r 's/(@@ )|(@@ ?$)//g'  $DATA/test.$tgt > $ref
  python -u $WORK_PATH/thumt/bin/translator.py \
    --models phase1transformer \
    --input $inp \
    --output $hyp \
    --vocabulary $DATA/dict $DATA/dict \
    --checkpoints $MODEL_PATH/eval --specific-ckpt $MODEL_PATH/eval/$model \
    --parameters=device_list=[$device],decode_batch_size=32,decode_alpha=0.6,beam_size=5,phase=1,ratio=0,flip_ratio=$flip_ratio,sample_infer_mode=0,reuse_encoder=True \
    --advice $DATA/test.advices 2>&1 | tee $MODEL_PATH/eval/test.log.$model.noad.$time

  sed -r 's/(@@ )|(@@ ?$)//g'  $hyp > $hyp.debpe
  sacrebleu $ref -i $hyp.debpe -tok none -m bleu -w 4 -b > $MODEL_PATH/eval/bleu.log.$model.noad.$time
done

