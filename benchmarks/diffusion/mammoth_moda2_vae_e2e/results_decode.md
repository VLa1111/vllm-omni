model=/root/models/MammothModa2-Preview subfolder=gen_vae.* (checkpoint shards) dtype=bf16
device=NVIDIA RTX PRO 6000 Blackwell Server Edition (97251 MiB total, 96329 MiB free before the sweep)
torch=2.13.0+cu130 diffusers=0.40.0
warmup=2 iters=5 seed=42 batch=1,2,4
tile_sample_min_size=1024 tile_latent_min_size=128 tile_overlap_factor=0.25 scale=8

| Size | Batch | Mode | Slicing | Tiling | Decode ms mean (min-max) | Peak MiB | PSNR dB | Max abs diff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024x1024 | 1 | baseline | no | no | 101.6 (101.6-101.6) | 2638 | - | - |
| 1024x1024 | 1 | slicing | no | no | 101.7 (101.6-101.9) | 2638 | identical | 0.0000 |
| 1024x1024 | 1 | tiling | no | no | 101.7 (101.6-101.7) | 2638 | identical | 0.0000 |
| 1024x1024 | 1 | slicing+tiling | no | no | 101.6 (101.6-101.7) | 2638 | identical | 0.0000 |
| 1536x1536 | 1 | baseline | no | no | 256.9 (256.8-257.2) | 5687 | - | - |
| 1536x1536 | 1 | slicing | no | no | 256.8 (256.8-256.9) | 5687 | identical | 0.0000 |
| 1536x1536 | 1 | tiling | no | yes | 345.1 (343.9-346.0) | 2647 | 55.77 | 0.0742 |
| 1536x1536 | 1 | slicing+tiling | no | yes | 345.3 (344.7-345.6) | 2647 | 55.77 | 0.0742 |
| 2048x2048 | 1 | baseline | no | no | 488.1 (487.9-488.3) | 9953 | - | - |
| 2048x2048 | 1 | slicing | no | no | 488.3 (488.2-488.5) | 9953 | identical | 0.0000 |
| 2048x2048 | 1 | tiling | no | yes | 745.7 (744.9-747.2) | 2678 | 54.37 | 0.0938 |
| 2048x2048 | 1 | slicing+tiling | no | yes | 745.2 (744.7-746.0) | 2678 | 54.37 | 0.0938 |
| 3072x3072 | 1 | baseline | no | no | 1366.0 (1365.8-1366.2) | 22146 | - | - |
| 3072x3072 | 1 | slicing | no | no | 1366.6 (1366.3-1367.2) | 22146 | identical | 0.0000 |
| 3072x3072 | 1 | tiling | no | yes | 1713.4 (1711.1-1716.5) | 2748 | 53.51 | 0.1123 |
| 3072x3072 | 1 | slicing+tiling | no | yes | 1714.3 (1710.3-1719.6) | 2748 | 53.51 | 0.1123 |
| 1024x1024 | 2 | baseline | no | no | 200.5 (200.4-200.6) | 4307 | - | - |
| 1024x1024 | 2 | slicing | yes | no | 204.1 (204.0-204.4) | 2650 | identical | 0.0000 |
| 1024x1024 | 2 | tiling | no | no | 200.5 (200.4-200.5) | 4307 | identical | 0.0000 |
| 1024x1024 | 2 | slicing+tiling | yes | no | 203.8 (203.8-203.9) | 2650 | identical | 0.0000 |
| 1536x1536 | 2 | baseline | no | no | 570.4 (569.7-571.0) | 9443 | - | - |
| 1536x1536 | 2 | slicing | yes | no | 515.3 (515.2-515.4) | 5714 | identical | 0.0000 |
| 1536x1536 | 2 | tiling | no | yes | 631.3 (630.6-631.7) | 4323 | 55.18 | 0.0825 |
| 1536x1536 | 2 | slicing+tiling | yes | yes | 689.9 (689.6-690.5) | 2674 | 55.18 | 0.0825 |
| 2048x2048 | 2 | baseline | no | no | 1134.7 (1134.1-1134.9) | 16634 | - | - |
| 2048x2048 | 2 | slicing | yes | no | 978.1 (977.9-978.2) | 10003 | identical | 0.0000 |
| 2048x2048 | 2 | tiling | no | yes | 1346.6 (1346.0-1347.1) | 4388 | 54.36 | 0.0996 |
| 2048x2048 | 2 | slicing+tiling | yes | yes | 1488.8 (1486.3-1490.8) | 2729 | 54.36 | 0.0996 |
| 3072x3072 | 2 | baseline | no | no | 2697.1 (2696.6-2697.8) | 44092 | - | - |
| 3072x3072 | 2 | slicing | yes | no | 2731.4 (2731.1-2731.7) | 22258 | identical | 0.0000 |
| 3072x3072 | 2 | tiling | no | yes | 3054.0 (3051.8-3055.8) | 4525 | 54.03 | 0.1123 |
| 3072x3072 | 2 | slicing+tiling | yes | yes | 3420.7 (3417.5-3424.4) | 2860 | 54.03 | 0.1123 |
| 1024x1024 | 4 | baseline | no | no | 359.3 (359.1-359.4) | 8416 | - | - |
| 1024x1024 | 4 | slicing | yes | no | 407.3 (407.2-407.4) | 2675 | identical | 0.0000 |
| 1024x1024 | 4 | tiling | no | no | 359.4 (359.3-359.5) | 8416 | identical | 0.0000 |
| 1024x1024 | 4 | slicing+tiling | yes | no | 407.5 (407.4-407.9) | 2675 | identical | 0.0000 |
| 1536x1536 | 4 | baseline | no | no | 1049.0 (1048.2-1049.5) | 16960 | - | - |
| 1536x1536 | 4 | slicing | yes | no | 1029.9 (1029.8-1030.2) | 5770 | identical | 0.0000 |
| 1536x1536 | 4 | tiling | no | yes | 1099.8 (1099.4-1100.3) | 8448 | 55.35 | 0.0938 |
| 1536x1536 | 4 | slicing+tiling | yes | yes | 1379.6 (1378.6-1380.3) | 2730 | 55.35 | 0.0938 |
| 2048x2048 | 4 | baseline | no | no | 2107.6 (2107.3-2108.1) | 28973 | - | - |
| 2048x2048 | 4 | slicing | yes | no | 1955.9 (1955.8-1956.1) | 10103 | identical | 0.0000 |
| 2048x2048 | 4 | tiling | no | yes | 2317.9 (2316.4-2319.6) | 8578 | 54.19 | 0.1087 |
| 2048x2048 | 4 | slicing+tiling | yes | yes | 2988.5 (2981.2-2991.7) | 2828 | 54.19 | 0.1087 |
| 3072x3072 | 4 | baseline | no | no | 40652.0 (30302.4-43240.7) | 73944 | - | - |
| 3072x3072 | 4 | slicing | yes | no | 5467.1 (5466.4-5468.2) | 22483 | 68.29 | 0.0156 |
| 3072x3072 | 4 | tiling | no | yes | 5256.3 (5253.5-5260.9) | 8852 | 54.14 | 0.1113 |
| 3072x3072 | 4 | slicing+tiling | yes | yes | 6837.0 (6828.9-6848.3) | 3085 | 54.14 | 0.1113 |
