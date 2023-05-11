import time

#out_dir = 'out-owt-gpt2mini'
out_dir = 'out-funcom-java-long-raw-jam350m-so'
eval_interval = 100
eval_iters = 80
wandb_log = True
wandb_project = 'funcom-java-long-raw'
wandb_run_name = 'ft-jam350m-so-1'

dataset = 'funcom-java-long_raw'
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

#n_layer = 24
#n_head = 16
#n_embd = 1024
#dropout = 0.2

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters

# funcom has 29,245,334 tokens

block_size = 256

batch_size = 4 #16
gradient_accumulation_steps = 32
#max_iters = 5600 # 172394 training samples

max_iters = 242000 + 3600 + 3600

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False
