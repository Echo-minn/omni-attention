## Attention patterns

### Document pattern
```shell
B: 1, H: 8, seq_len: 512, head_dim: 64, Q_BLOCK_SIZE: 64, KV_BLOCK_SIZE: 64
Document segments: 256, 68, 188
Saving to data/attn_data/512_document_64_64.pt...
Sparsity: 0.5975 (59.75% sparse)
Block mask pattern (head=0, F=FULL, C=CAUSAL, P=PARTIAL):
  q_block=0: F F F F        
  q_block=1: F F F F        
  q_block=2: F F F F        
  q_block=3: F F F F        
  q_block=4:         F P    
  q_block=5:         P P P P
  q_block=6:           P F F
  q_block=7:           P F F
```

### Interleaved pattern
```shell
B: 1, H: 8, seq_len: 512, head_dim: 64, Q_BLOCK_SIZE: 64, KV_BLOCK_SIZE: 64
Segment lengths: 133, 309, 70

Saving to data/attn_data/512_interleaved_64_64.pt...
✓ Saved. File size: 16.29 MB
Sparsity: 0.3175 (31.75% sparse)
Block mask pattern (head=0, F=FULL, C=CAUSAL, P=PARTIAL):
  q_block=0: C              
  q_block=1: F C            
  q_block=2: F F P P P P P  
  q_block=3: F F F F F F P  
  q_block=4: F F F F F F P  
  q_block=5: F F F F F F P  
  q_block=6: F F F F F F P  
  q_block=7: F F F F F F F C
```

### Causal pattern
```shell
B: 1, H: 8, seq_len: 512, head_dim: 64, Q_BLOCK_SIZE: 64, KV_BLOCK_SIZE: 64
Generating fixed debug data: B=1, H=8, seq_len=512, head_dim=64, Q_BLOCK_SIZE=64, KV_BLOCK_SIZE=64

Saving to data/attn_data/512_causal_64_64.pt...
✓ Saved. File size: 16.01 MB
Sparsity: 0.4990 (49.90% sparse)
Block mask pattern (head=0, F=FULL, C=CAUSAL, P=PARTIAL):
  q_block=0: C              
  q_block=1: F C            
  q_block=2: F F C          
  q_block=3: F F F C        
  q_block=4: F F F F C      
  q_block=5: F F F F F C    
  q_block=6: F F F F F F C  
  q_block=7: F F F F F F F C
```
## Arbitrary mask

### 1024_128_128_F_P.pt
```shell
random FULL/CAUSAL debug data: B=1, H=8, seq_len=1024, head_dim=64, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128

Saving to data/1024/128_128_F_P.pt...
✓ Saved. File size: 52.01 MB
Sparsity: 0.4375 (43.75% sparse)
Block mask pattern (head=0, F=FULL, C=CAUSAL, P=PARTIAL):
  q_block=0: F
  q_block=1: P F
  q_block=2: F P F
  q_block=3: F F F F
  q_block=4: P F P P F
  q_block=5: F P F F F F
  q_block=6: F P P F F F F
  q_block=7: F P F P F F F F

============================================================
Testing with data: data/1024/128_128_F_P.pt ...
  Q shape: torch.Size([1, 8, 1024, 64]); Sparsity: 0.4375; Pattern: random_full_partial
============================================================
  Simple FAILED❌, time: 13.91ms, TFLOPS: 0.16, speedup: 0.02x
  Prefetch FAILED❌, time: 0.41ms, TFLOPS: 5.31, speedup: 0.77x
  Shared_kv PASSED✅, time: 0.23ms, TFLOPS: 9.37, speedup: 1.37x
  Swizzle PASSED✅, time: 0.18ms, TFLOPS: 12.05, speedup: 1.76x
  Flex PASSED✅, time: 0.32ms, TFLOPS: 6.86
```

### 1024_64_64_F_P.pt
```shell
random FULL/PARTIAL debug data: B=1, H=8, seq_len=1024, head_dim=64, Q_BLOCK_SIZE=64, KV_BLOCK_SIZE=64

Saving to data/1024/64_64_F_P.pt...
✓ Saved. File size: 52.02 MB
Sparsity: 0.4702 (47.02% sparse)
Block mask pattern (head=0, F=FULL, C=CAUSAL, P=PARTIAL):
  q_block=0: F
  q_block=1: P F
  q_block=2: P P F
  q_block=3: P F F F
  q_block=4: P F P P P
  q_block=5: F P F P F F
  q_block=6: F P P F F F F
  q_block=7: F P F P F P F F
  q_block=8: P F P P P F P P F
  q_block=9: F F P F F F P F F P
  q_block=10: F P F F P F P P P P F
  q_block=11: P P F F F P P P P F F F
  q_block=12: F P F F P F F P P F P F F
  q_block=13: P P F F F P F P F F P P F F
  q_block=14: F P F P F P P F F F P F P P F
  q_block=15: P P P F F F P F F P P P P F P P


============================================================
Testing with data: data/1024/64_64_F_P.pt ...
  Q shape: torch.Size([1, 8, 1024, 64]); Sparsity: 0.4692; Pattern: random_full_partial
============================================================
  Simple FAILED❌, time: 12.16ms, TFLOPS: 0.18, speedup: 0.03x
  Prefetch FAILED❌, time: 0.42ms, TFLOPS: 5.20, speedup: 0.76x
  Shared_kv PASSED✅, time: 0.20ms, TFLOPS: 10.76, speedup: 1.58x
  Swizzle PASSED✅, time: 0.15ms, TFLOPS: 14.17, speedup: 2.08x
  Flex PASSED✅, time: 0.32ms, TFLOPS: 6.80
```

### 1024_128_128_F_C_P.pt
```shell
On random FULL/CAUSAL/PARTIAL debug data: B=1, H=8, seq_len=1024, head_dim=64, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128

Saving to data/1024/128_128_F_C_P.pt...
✓ Saved. File size: 53.63 MB
Sparsity: 0.4937 (49.37% sparse)
Block mask pattern (head=0, F=FULL, C=CAUSAL, P=PARTIAL):
  q_block=0: C
  q_block=1: P C
  q_block=2: P P C
  q_block=3: C F F C
  q_block=4: P F P P P
  q_block=5: C P F P F C
  q_block=6: F P P C F C C
  q_block=7: F P C P F C C C

============================================================
Testing with data: data/1024/128_128_F_C_P.pt ...
  Q shape: torch.Size([1, 8, 1024, 64]); Sparsity: 0.4937; Pattern: random_full_causal_partial
============================================================
  Simple PASSED✅, time: 15.99ms, TFLOPS: 0.14, speedup: 0.02x
  Prefetch FAILED❌, time: 0.53ms, TFLOPS: 4.11, speedup: 0.60x
  Shared_kv FAILED❌, time: 0.22ms, TFLOPS: 9.80, speedup: 1.43x
  Swizzle FAILED❌, time: 0.17ms, TFLOPS: 12.79, speedup: 1.86x
  Flex PASSED✅, time: 0.32ms, TFLOPS: 6.86
```

### 1024_64_64_F_C_P.pt
```shell
On random FULL/CAUSAL/PARTIAL debug data: B=1, H=8, seq_len=1024, head_dim=64, Q_BLOCK_SIZE=64, KV_BLOCK_SIZE=64

Saving to data/1024/64_64_F_C_P.pt...
✓ Saved. File size: 53.53 MB
Sparsity: 0.4952 (49.52% sparse)
Block mask pattern (head=0, F=FULL, C=CAUSAL, P=PARTIAL):
  q_block=0: C
  q_block=1: P C
  q_block=2: P P C
  q_block=3: C F F C
  q_block=4: P F P P P
  q_block=5: C P F P F C
  q_block=6: F P P C F C C
  q_block=7: F P C P F C C C
  q_block=8: P F P C C C P P C
  q_block=9: F F P C F F P F F P
  q_block=10: C C C F C F C P P P C
  q_block=11: P P C C F C C P P F F C
  q_block=12: C P C F C C F P P F P F C
  q_block=13: P P F F F C F P C C P P F C
  q_block=14: F C F P F P P F F F P C P P C
  q_block=15: P P C F F F P F C C P C P F C P

============================================================
Testing with data: data/1024/64_64_F_C_P.pt ...
  Q shape: torch.Size([1, 8, 1024, 64]); Sparsity: 0.4952; Pattern: random_full_causal_partial
============================================================
  Simple PASSED✅, time: 16.18ms, TFLOPS: 0.14, speedup: 0.02x
  Prefetch FAILED❌, time: 0.49ms, TFLOPS: 4.50, speedup: 0.60x
  Shared_kv FAILED❌, time: 0.21ms, TFLOPS: 10.64, speedup: 1.43x
  Swizzle FAILED❌, time: 0.16ms, TFLOPS: 13.70, speedup: 1.84x
  Flex PASSED✅, time: 0.29ms, TFLOPS: 7.46
```

### document mask
| seq_len | prefetch | shared_kv | swizzle |
|---------|----------|----------|---------|
| 512 - 64     |   TFLOPS: 1.45, speedup: 0.74x   |   TFLOPS: 5.03, speedup: 2.57x     |     TFLOPS: 5.25, speedup: 2.68x    |
| 512 - 128    |   TFLOPS: 1.36, speedup: 0.68x   |   TFLOPS: 4.58, speedup: 2.27x     |     TFLOPS: 5.00, speedup: 2.48x    |
| 1024 - 64    |   TFLOPS: 4.78, speedup: 0.61x   |   TFLOPS: 13.40, speedup: 1.70x    |     TFLOPS: 16.91, speedup: 2.14x   |
| 1024 - 128   |   TFLOPS: 4.81, speedup: 0.60x   |   TFLOPS: 13.81, speedup: 1.72x    |     TFLOPS: 17.19, speedup: 2.14x   |
| 2048 - 64    |   TFLOPS: 11.92, speedup: 0.42x  |   TFLOPS: 20.07, speedup: 0.71x    |     TFLOPS: 29.76, speedup: 1.05x   |
| 2048 - 128   |   TFLOPS: 11.50, speedup: 0.48x  |   TFLOPS: 18.43, speedup: 0.77x    |     TFLOPS: 24.87, speedup: 1.04x   |


### interleaved mask
| seq_len | prefetch | shared_kv | swizzle |
|---------|------|--------|---------|
| 512 - 64    |   TFLOPS: 1.38, speedup: 1.02x   |   TFLOPS: 4.11, speedup: 3.03x    |     TFLOPS: 4.62, speedup: 3.41x    |
| 512 - 128   |   TFLOPS: 1.09, speedup: 0.55x   |   TFLOPS: 3.04, speedup: 1.55x    |     TFLOPS: 3.38, speedup: 1.72x    |
| 1024 - 64   |   TFLOPS: 4.33, speedup: 0.54x   |   TFLOPS: 9.31, speedup: 1.15x    |     TFLOPS: 13.19, speedup: 1.64x   |
| 1024 - 128  |   TFLOPS: 4.73, speedup: 0.71x   |   TFLOPS: 9.84, speedup: 1.48x    |     TFLOPS: 12.24, speedup: 1.84x   |
| 2048 - 64   |   TFLOPS: 9.69, speedup: 0.39x   |   TFLOPS: 18.04, speedup: 0.73x   |     TFLOPS: 28.17, speedup: 1.14x   |
| 2048 - 128  |   TFLOPS: 14.26, speedup: 0.48x  |   TFLOPS: 23.48, speedup: 0.79x   |     TFLOPS: 33.60, speedup: 1.14x   |

### causal mask
| seq_len | prefetch | shared_kv | swizzle |
|---------|------|--------|---------|
| 512 - 64    |   TFLOPS: 1.49, speedup: 0.79x   |   TFLOPS: 4.21, speedup: 2.23x     |     TFLOPS: 4.57, speedup: 2.42x    |
| 512 - 128   |   TFLOPS: 1.44, speedup: 0.67x   |   TFLOPS: 3.64, speedup: 1.68x     |     TFLOPS: 4.35, speedup: 2.01x    |
| 1024 - 64   |   TFLOPS: 4.78, speedup: 0.64x   |   TFLOPS: 9.25, speedup: 1.23x     |     TFLOPS: 14.39, speedup: 1.92x   |
| 1024 - 128  |   TFLOPS: 5.06, speedup: 0.68x   |   TFLOPS: 9.81, speedup: 1.31x     |     TFLOPS: 12.47, speedup: 1.67x   |
| 2048 - 64   |   TFLOPS: 12.64, speedup: 0.44x  |   TFLOPS: 18.95, speedup: 0.66x    |     TFLOPS: 31.69, speedup: 1.10x   |
| 2048 - 128  |   TFLOPS: 13.62, speedup: 0.43x  |   TFLOPS: 23.42, speedup: 0.74x    |     TFLOPS: 32.27, speedup: 1.02x   |

### F_C_P arbitrary combined mask
| seq_len | prefetch | shared_kv | swizzle |
|---------|----------|-----------|---------|
| 512 - 64    |   TFLOPS: 1.29, speedup: 0.65x   |   TFLOPS: 4.27, speedup: 2.15x     |    TFLOPS: 4.76, speedup: 2.40x     |
| 512 - 128   |   TFLOPS: 1.15, speedup: 0.80x   |   TFLOPS: 3.49, speedup: 2.43x     |    TFLOPS: 4.41, speedup: 3.08x     |
| 1024 - 64   |   TFLOPS: 4.45, speedup: 0.49x   |    TFLOPS: 10.39, speedup: 1.14x    |   TFLOPS: 11.40, speedup: 1.25x      |
| 1024 - 128  |   TFLOPS: 4.44, speedup: 0.60x   |     TFLOPS: 9.36, speedup: 1.27x   |    TFLOPS: 10.68, speedup: 1.45x     |
| 2048 - 64   |   TFLOPS: 12.03, speedup: 0.47x   |    TFLOPS: 19.86, speedup: 0.78x    |    TFLOPS: 29.88, speedup: 1.17x     |
| 2048 - 128  |   TFLOPS: 13.08, speedup: 0.54x   |    TFLOPS: 20.83, speedup: 0.87x    |    TFLOPS: 26.77, speedup: 1.11x     |

### F_P arbitrary combined mask
| seq_len | prefetch | shared_kv | swizzle |
|---------|----------|-----------|---------|
| 512 - 64    |   TFLOPS: 1.26, speedup: 0.64x   |    TFLOPS: 3.40, speedup: 1.74x    |    TFLOPS: 3.49, speedup: 1.78x     |
| 512 - 128   |   TFLOPS: 1.25, speedup: 0.93x   |    TFLOPS: 2.78, speedup: 2.08x    |    TFLOPS: 3.84, speedup: 2.87x     |
| 1024 - 64   |   TFLOPS: 5.48, speedup: 0.69x   |    TFLOPS: 9.64, speedup: 1.21x    |    TFLOPS: 14.10, speedup: 1.78x    |
| 1024 - 128  |   TFLOPS: 4.78, speedup: 0.74x   |    TFLOPS: 10.47, speedup: 1.63x   |    TFLOPS: 13.64, speedup: 2.12x    |
| 2048 - 64   |   TFLOPS: 14.85, speedup: 0.53x  |    TFLOPS: 21.84, speedup: 0.78x   |    TFLOPS: 33.73, speedup: 1.20x    |
| 2048 - 128  |   TFLOPS: 15.64, speedup: 0.53x  |    TFLOPS: 23.20, speedup: 0.78x   |    TFLOPS: 33.57, speedup: 1.13x    |