# Profiling data

memory allocation, TFLOPS, diff, speedup, time

## test logs

```shell
(779) (base) mira@ip-172-31-15-66:/opt/dlami/nvme/mira/omni-attention/omni-attn-mma$ python test_with_debug_data.py --input debug_data_partial.pt 
Loading debug data from debug_data_partial.pt...

Testing with fixed data:
  {'B': 1, 'H': 8, 'seq_len': 512, 'head_dim': 64, 'BLOCK_SIZE': 128, 'pattern': 'random_full_causal_partial'}
  Q shape: torch.Size([1, 8, 512, 64]), dtype: torch.float16
  Reference output shape: torch.Size([1, 8, 512, 64])

============================================================
Testing omni_attention_simple...
============================================================
has_partial
has_partial
[omni_attention_simple] PASSED (max_diff=0.000488, rel_diff=40.009911)
  Time: 25.56ms

============================================================
Testing flex_attention...
============================================================
[flex_attention] PASSED (max_diff=0.000488, rel_diff=65.018341)
  Time: 2.54ms

============================================================
Comparing omni_attention_simple vs flex_attention...
============================================================
[omni vs flex] PASSED (max_diff=0.000488, rel_diff=91.895615)
  Speedup: 10.07x (omni faster)
    omni: 25.56ms
    flex: 2.54ms
```

```shell
(779) (base) mira@ip-172-31-15-66:/opt/dlami/nvme/mira/omni-attention/omni-attn-mma$ python test_with_debug_data.py --input data/1024/debug_data_F_P.pt 
Loading debug data from data/1024/debug_data_F_P.pt...

Testing with fixed data:
  {'B': 1, 'H': 8, 'seq_len': 512, 'head_dim': 64, 'Q_BLOCK_SIZE': 128, 'KV_BLOCK_SIZE': 128, 'seed': 40, 'device': 'cuda', 'pattern': 'random_full_causal'}
  Q shape: torch.Size([1, 8, 512, 64]), dtype: torch.float16; Q_BLOCK_SIZE: 128, KV_BLOCK_SIZE: 128
  Reference output shape: torch.Size([1, 8, 512, 64])

============================================================
Testing flex_attention...
============================================================
[flex_attention] PASSED (max_diff=0.000732, mean_diff=0.000032), min_diff=0.000000
  Time: 0.27ms

============================================================
Testing omni_attention_simple...
============================================================
has_partial
has_partial
[omni_attention_simple] PASSED (max_diff=0.000732, mean_diff=0.000029), min_diff=0.000000
  Time: 10.74ms

============================================================
Testing omni_attn_shared_kv...
============================================================
[omni_attn_shared_kv] PASSED (max_diff=0.000732, mean_diff=0.000032), min_diff=0.000000
  Time: 0.17ms

============================================================
Comparing omni_attention_simple vs flex_attention...
============================================================
[omni vs flex] PASSED (max_diff=0.000488, mean_diff=0.000006), min_diff=0.000000
  Speedup: 1.58x (omni faster)
    omni: 0.17ms
    flex: 0.27ms
```

```shell
(779) (base) mira@ip-172-31-15-66:/opt/dlami/nvme/mira/omni-attention/omni-attn-mma$ python test_with_debug_data.py --input data/1024/debug_data_F.pt 
Loading debug data from data/1024/debug_data_F.pt...

Testing with fixed data:
  {'B': 1, 'H': 8, 'seq_len': 1024, 'head_dim': 64, 'Q_BLOCK_SIZE': 128, 'KV_BLOCK_SIZE': 128, 'seed': 42, 'device': 'cuda', 'pattern': 'random_full_causal'}
  Q shape: torch.Size([1, 8, 1024, 64]), dtype: torch.float16; Q_BLOCK_SIZE: 128, KV_BLOCK_SIZE: 128
  Reference output shape: torch.Size([1, 8, 1024, 64])

============================================================
Testing flex_attention...
============================================================
[flex_attention] PASSED (max_diff=0.000488, mean_diff=0.000025), min_diff=0.000000
  Time: 0.92ms

============================================================
Testing omni_attention_simple...
============================================================
[omni_attention_simple] PASSED (max_diff=0.000488, mean_diff=0.000022), min_diff=0.000000
  Time: 21.87ms

============================================================
Testing omni_attn_shared_kv...
============================================================
[omni_attn_shared_kv] PASSED (max_diff=0.000488, mean_diff=0.000025), min_diff=0.000000
  Time: 0.28ms

============================================================
Comparing omni_attention_simple vs flex_attention...
============================================================
[omni vs flex] PASSED (max_diff=0.000488, mean_diff=0.000005), min_diff=0.000000
  Speedup: 3.28x (omni faster)
    omni: 0.28ms
    flex: 0.92ms
```