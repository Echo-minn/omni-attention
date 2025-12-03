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