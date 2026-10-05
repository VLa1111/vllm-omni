model=/root/models/MammothModa2-Preview subfolder=gen_vae.* (checkpoint shards) dtype=bf16
device=NVIDIA RTX PRO 6000 Blackwell Server Edition (97251 MiB total, 96329 MiB free before the sweep)
torch=2.13.0+cu130 diffusers=0.40.0
warmup=2 iters=5 seed=42 batch=1,2,4
tile_sample_min_size=1024 tile_latent_min_size=128 tile_overlap_factor=0.25 scale=8

| Size | Batch | Mode | Slicing | Tiling | Decode ms | Peak MiB | PSNR dB | Max abs diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024x1024 | 1 | baseline | no | no | 101.7 | 2638 | - | - |
| 1024x1024 | 1 | slicing | no | no | 101.6 | 2638 | identical | 0.0000 |
| 1024x1024 | 1 | tiling | no | no | 101.6 | 2638 | identical | 0.0000 |
| 1024x1024 | 1 | slicing+tiling | no | no | 101.6 | 2638 | identical | 0.0000 |
| 1536x1536 | 1 | baseline | no | no | 256.8 | 5687 | - | - |
| 1536x1536 | 1 | slicing | no | no | 256.7 | 5687 | identical | 0.0000 |
| 1536x1536 | 1 | tiling | no | yes | 343.4 | 2647 | 55.77 | 0.0742 |
| 1536x1536 | 1 | slicing+tiling | no | yes | 343.4 | 2647 | 55.77 | 0.0742 |
| 2048x2048 | 1 | baseline | no | no | 487.8 | 9953 | - | - |
| 2048x2048 | 1 | slicing | no | no | 488.0 | 9953 | identical | 0.0000 |
| 2048x2048 | 1 | tiling | no | yes | 743.5 | 2678 | 54.37 | 0.0938 |
| 2048x2048 | 1 | slicing+tiling | no | yes | 748.5 | 2678 | 54.37 | 0.0938 |
| 3072x3072 | 1 | baseline | no | no | 1759.5 | 22146 | - | - |
| 3072x3072 | 1 | slicing | no | no | 1366.0 | 22146 | identical | 0.0000 |
| 3072x3072 | 1 | tiling | no | yes | 1709.8 | 2748 | 53.51 | 0.1123 |
| 3072x3072 | 1 | slicing+tiling | no | yes | 1710.1 | 2748 | 53.51 | 0.1123 |
| 1024x1024 | 2 | baseline | no | no | 200.4 | 4307 | - | - |
| 1024x1024 | 2 | slicing | yes | no | 203.8 | 2650 | identical | 0.0000 |
| 1024x1024 | 2 | tiling | no | no | 200.4 | 4307 | identical | 0.0000 |
| 1024x1024 | 2 | slicing+tiling | yes | no | 203.9 | 2650 | identical | 0.0000 |
| 1536x1536 | 2 | baseline | no | no | 570.3 | 9443 | - | - |
| 1536x1536 | 2 | slicing | yes | no | 515.0 | 5714 | identical | 0.0000 |
| 1536x1536 | 2 | tiling | no | yes | 630.5 | 4323 | 55.18 | 0.0825 |
| 1536x1536 | 2 | slicing+tiling | yes | yes | 688.9 | 2674 | 55.18 | 0.0825 |
| 2048x2048 | 2 | baseline | no | no | 1134.6 | 16634 | - | - |
| 2048x2048 | 2 | slicing | yes | no | 977.5 | 10003 | identical | 0.0000 |
| 2048x2048 | 2 | tiling | no | yes | 1344.5 | 4388 | 54.36 | 0.0996 |
| 2048x2048 | 2 | slicing+tiling | yes | yes | 1485.6 | 2729 | 54.36 | 0.0996 |
| 3072x3072 | 2 | baseline | no | no | 2695.7 | 44092 | - | - |
| 3072x3072 | 2 | slicing | yes | no | 2730.7 | 22258 | identical | 0.0000 |
| 3072x3072 | 2 | tiling | no | yes | 3053.2 | 4525 | 54.03 | 0.1123 |
| 3072x3072 | 2 | slicing+tiling | yes | yes | 3419.0 | 2860 | 54.03 | 0.1123 |
| 1024x1024 | 4 | baseline | no | no | 359.1 | 8416 | - | - |
| 1024x1024 | 4 | slicing | yes | no | 407.2 | 2675 | identical | 0.0000 |
| 1024x1024 | 4 | tiling | no | no | 359.1 | 8416 | identical | 0.0000 |
| 1024x1024 | 4 | slicing+tiling | yes | no | 407.3 | 2675 | identical | 0.0000 |
| 1536x1536 | 4 | baseline | no | no | 1048.3 | 16960 | - | - |
| 1536x1536 | 4 | slicing | yes | no | 1029.3 | 5770 | identical | 0.0000 |
| 1536x1536 | 4 | tiling | no | yes | 1099.0 | 8448 | 55.35 | 0.0938 |
| 1536x1536 | 4 | slicing+tiling | yes | yes | 1377.0 | 2730 | 55.35 | 0.0938 |
| 2048x2048 | 4 | baseline | no | no | 2108.4 | 28973 | - | - |
| 2048x2048 | 4 | slicing | yes | no | 1955.3 | 10103 | identical | 0.0000 |
| 2048x2048 | 4 | tiling | no | yes | 2317.7 | 8578 | 54.19 | 0.1087 |
| 2048x2048 | 4 | slicing+tiling | yes | yes | 2985.0 | 2828 | 54.19 | 0.1087 |
| 3072x3072 | 4 | baseline | no | no | 40649.7 | 73944 | - | - |
| 3072x3072 | 4 | slicing | yes | no | 5464.1 | 22483 | 68.29 | 0.0156 |
| 3072x3072 | 4 | tiling | no | yes | 5252.4 | 8852 | 54.14 | 0.1113 |
| 3072x3072 | 4 | slicing+tiling | yes | yes | 6834.2 | 3085 | 54.14 | 0.1113 |
