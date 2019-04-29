
| option      | description                                             | example                                    |
|-------------|---------------------------------------------------------|--------------------------------------------|
| -arae       | arae setting. w/o the flag will be a basic transformer  | -arae                                      |
| -save_gan   | path where to save model                                | -save_gan ./data_autoenc/gan               |
| -model_arae | path to arae model from where to load it                | -model_arae data_autoenc/gan_step_17000.pt |

Examples of scripts:

TRAIN:

```console
python3.6  train.py -data ./data/demo -save_model ./data_autoenc/model -save_gan ./data_autoenc/gan \
         -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  \
         -encoder_type transformer -decoder_type transformer -position_encoding \
         -train_steps 200000  -max_generator_batches 2 -dropout 0.1 \
         -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 \
         -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 \
         -max_grad_norm 0 -param_init 0  -param_init_glorot \
         -label_smoothing 0.1 -valid_steps 1000 -save_checkpoint_steps 1000 \
         -world_size 1 -gpu_ranks 0 -arae \
         -train_from ./data_autoenc/model_step_17000.pt -model_arae data_autoenc/gan_step_17000.pt
```

GENERATE:

```console
python3.6 generate.py -model data_autoenc/model_step_17000.pt -model_arae data_autoenc/gan_step_17000.pt -src data_snli/src-val.txt -output pred.txt -arae
```

ADVICE FOR RUNNING TRAINING (with tee):

```console
CUDA_VISIBLE_DEVICES=4 sh_scripts/run_train.sh > >(tee stdout.log) 2> >(tee stderr.log >&2)
```
