# -*- coding: utf-8 -*-
MAX_LEN = 50
neg_table_size = 10000000
NEG_SAMPLE_POWER = 0.75
init_batch_size = 512 # set your batch size
all_batch_size = init_batch_size
song_batch_size = all_batch_size // 2
num_epoch = 20
embed_size = 200  # dimension of embedding
text_embed_size = embed_size // 2
structure_embed_size = embed_size - text_embed_size
lr = 1e-3
CUDA_VISIBLE_DEVICES = '0'