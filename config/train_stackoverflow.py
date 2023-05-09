import time

#out_dir = 'out-owt-gpt2mini'
out_dir = 'out-stackoverflow_scratch_gpt2med_iter300k'
eval_interval = 1000
eval_iters = 40
wandb_log = False # feel free to turn on
wandb_project = 'stackoverflow'
wandb_run_name = 'ft-gpt2-300k' #+ str(time.time())

dataset = 'stackoverflow'
init_from = 'scratch'
#init_from = 'gpt2-large'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

#n_layer = 6
#n_head = 6
#n_embd = 384
#dropout = 0.2

block_size = 256

# gpt2-large
#n_layer = 36
#n_head = 20
#n_embd = 1280
#dropout = 0.2

# gpt2-medium
n_layer = 24
n_head = 16
n_embd = 1024
dropout = 0.2

#n_layer = 32
#n_head = 32
#n_embd = 1536
#dropout = 0.2

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters

# stackoverflow has 10,495,518,108 tokens
# openwebtext has 9,035,582,489 tokens
# funcom_raw has 8,752,695,577 tokens

batch_size = 4 #16
gradient_accumulation_steps = 32
max_iters = 300000

#learning_rate = 3e-5
weight_decay = 1e-1
#decay_lr = False

# lora parameters, 0s mean disable
lora_rank = 0
lora_alpha = 0
lora_dropout = 0


