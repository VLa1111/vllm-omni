## Output consistency (req1, seed 42)

| Arm | Group | md5 | vs baseline | PSNR dB | max abs diff | byte-identical |
| --- | --- | --- | --- | ---: | ---: | --- |
| fp8_noffload | fp8 | a061cbb44929ee58ad650604b733d88f | (baseline) | - | - | - |
| fp8_model | fp8 | 3a47a511976e7a55eeb29b0613c3fc84 | fp8_noffload | 49.72 | 0.1922 | no |
| fp8_layerwise | fp8 | 3a47a511976e7a55eeb29b0613c3fc84 | fp8_noffload | 49.72 | 0.1922 | no |
| fp8_bothflags | fp8 | 3a47a511976e7a55eeb29b0613c3fc84 | fp8_noffload | 49.72 | 0.1922 | no |
| fp8_dlo_n0 | fp8 | 3a47a511976e7a55eeb29b0613c3fc84 | fp8_noffload | 49.72 | 0.1922 | no |
| fp8_dlo_n4 | fp8 | 3a47a511976e7a55eeb29b0613c3fc84 | fp8_noffload | 49.72 | 0.1922 | no |
| bf16_model | bf16 | 9add03c40f7cfaa1eb1ec49fac91ead0 | (baseline) | - | - | - |
| bf16_layerwise | bf16 | 9add03c40f7cfaa1eb1ec49fac91ead0 | bf16_model | inf (identical) | 0.0000 | yes |
| dlo_n0 | bf16 | 9add03c40f7cfaa1eb1ec49fac91ead0 | bf16_model | inf (identical) | 0.0000 | yes |
| dlo_n4 | bf16 | - | - | - | - | no output |
| dlo_n4_retry1 | bf16 | 9add03c40f7cfaa1eb1ec49fac91ead0 | bf16_model | inf (identical) | 0.0000 | yes |
| dlo_n4_retry2 | bf16 | 9add03c40f7cfaa1eb1ec49fac91ead0 | bf16_model | inf (identical) | 0.0000 | yes |

## Activation evidence (cleaned from serve logs)

### fp8_noffload

```text
[0;36m(DiffusionWorker pid=2867)[0;0m [32mINFO[0m [90m10-09 14:52:59[0m [90m[plain_fp8.py:191][0m Unpacked 318 torchao Float8Tensor linear(s) into plain (fp8 weight, fp32 row-scale) parameters
```

### fp8_model

```text
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:53:41[0m [90m[diffusers_loader.py:711][0m Online quantization with CPU offload, using cuda for weight loading (will offload back to CPU)
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:53:47[0m [90m[diffusers_loader.py:510][0m Stream-offloaded 0 online-quantized layers to CPU during weight loading
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:53:47[0m [90m[plain_fp8.py:191][0m Unpacked 318 torchao Float8Tensor linear(s) into plain (fp8 weight, fp32 row-scale) parameters
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:53:52[0m [90m[diffusers_loader.py:887][0m Quantization complete, offloaded model back to CPU
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:53:53[0m [90m[__init__.py:179][0m Enabling offloader backend: ModelLevelOffloadBackend
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:53:53[0m [90m[sequential_backend.py:382][0m Model-level offloading enabled: transformer <-> mllm (mutual exclusion)
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:54:24[0m [90m[sequential_backend.py:408][0m Model-level offloading disabled
```

### fp8_layerwise

```text
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:36[0m [90m[diffusers_loader.py:711][0m Online quantization with CPU offload, using cuda for weight loading (will offload back to CPU)
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:42[0m [90m[diffusers_loader.py:510][0m Stream-offloaded 0 online-quantized layers to CPU during weight loading
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:42[0m [90m[plain_fp8.py:191][0m Unpacked 318 torchao Float8Tensor linear(s) into plain (fp8 weight, fp32 row-scale) parameters
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:46[0m [90m[diffusers_loader.py:887][0m Quantization complete, offloaded model back to CPU
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:47[0m [90m[__init__.py:179][0m Enabling offloader backend: LayerWiseOffloadBackend
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:48[0m [90m[layerwise_backend.py:598][0m layerwise offload timing instrumentation enabled (VLLM_OMNI_OFFLOAD_TIMING=1)
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:50[0m [90m[layerwise_backend.py:614][0m Layer-wise offloading enabled on 40 layers (blocks)
0%|          | 0/2 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:56[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=452.4 ms (8813.56 MiB), compute=3136.3 ms, exposed_stall=182.7 ms, wall=3303.6 ms, hidden_h2d=269.7 ms (60% of h2d hidden)
50%|█████     | 1/2 [00:04<00:04,  4.37s/it][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:56[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.0 ms (8813.56 MiB), compute=250.8 ms, exposed_stall=199.6 ms, wall=435.0 ms, hidden_h2d=250.4 ms (56% of h2d hidden)
0%|          | 0/10 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:01[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.7 ms (8813.56 MiB), compute=1372.4 ms, exposed_stall=291.6 ms, wall=1645.9 ms, hidden_h2d=160.1 ms (35% of h2d hidden)
10%|█         | 1/10 [00:01<00:15,  1.68s/it][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:01[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.3 ms (8813.56 MiB), compute=136.2 ms, exposed_stall=315.5 ms, wall=433.7 ms, hidden_h2d=135.8 ms (30% of h2d hidden)
20%|██        | 2/10 [00:02<00:07,  1.03it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:01[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.3 ms (8813.56 MiB), compute=136.2 ms, exposed_stall=315.6 ms, wall=433.7 ms, hidden_h2d=135.8 ms (30% of h2d hidden)
30%|███       | 3/10 [00:02<00:05,  1.35it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:02[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.2 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=315.1 ms, wall=433.3 ms, hidden_h2d=136.1 ms (30% of h2d hidden)
40%|████      | 4/10 [00:03<00:03,  1.58it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:02[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.6 ms (8813.56 MiB), compute=136.9 ms, exposed_stall=314.1 ms, wall=432.8 ms, hidden_h2d=136.5 ms (30% of h2d hidden)
50%|█████     | 5/10 [00:03<00:02,  1.74it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:03[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.9 ms (8813.56 MiB), compute=136.9 ms, exposed_stall=314.4 ms, wall=433.3 ms, hidden_h2d=136.5 ms (30% of h2d hidden)
60%|██████    | 6/10 [00:04<00:02,  1.86it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:03[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.3 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=314.2 ms, wall=432.7 ms, hidden_h2d=136.1 ms (30% of h2d hidden)
70%|███████   | 7/10 [00:04<00:01,  1.94it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:04[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.7 ms (8813.56 MiB), compute=136.8 ms, exposed_stall=314.3 ms, wall=433.1 ms, hidden_h2d=136.4 ms (30% of h2d hidden)
80%|████████  | 8/10 [00:04<00:00,  2.00it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:04[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=452.3 ms (8813.56 MiB), compute=137.2 ms, exposed_stall=315.5 ms, wall=434.7 ms, hidden_h2d=136.9 ms (30% of h2d hidden)
90%|█████████ | 9/10 [00:05<00:00,  2.04it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:05[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.5 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=315.4 ms, wall=433.5 ms, hidden_h2d=136.1 ms (30% of h2d hidden)
0%|          | 0/10 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:05[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.4 ms (8813.56 MiB), compute=136.8 ms, exposed_stall=314.0 ms, wall=432.8 ms, hidden_h2d=136.4 ms (30% of h2d hidden)
10%|█         | 1/10 [00:00<00:04,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:06[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.4 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=314.4 ms, wall=432.8 ms, hidden_h2d=136.1 ms (30% of h2d hidden)
20%|██        | 2/10 [00:00<00:03,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:06[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.4 ms (8813.56 MiB), compute=136.8 ms, exposed_stall=313.9 ms, wall=432.6 ms, hidden_h2d=136.5 ms (30% of h2d hidden)
30%|███       | 3/10 [00:01<00:03,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:07[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.6 ms (8813.56 MiB), compute=136.9 ms, exposed_stall=314.0 ms, wall=432.9 ms, hidden_h2d=136.5 ms (30% of h2d hidden)
40%|████      | 4/10 [00:01<00:02,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:07[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.8 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=314.7 ms, wall=433.1 ms, hidden_h2d=136.1 ms (30% of h2d hidden)
50%|█████     | 5/10 [00:02<00:02,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:08[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.1 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=314.9 ms, wall=433.1 ms, hidden_h2d=136.1 ms (30% of h2d hidden)
60%|██████    | 6/10 [00:02<00:01,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:08[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.5 ms (8813.56 MiB), compute=136.4 ms, exposed_stall=314.3 ms, wall=432.4 ms, hidden_h2d=136.2 ms (30% of h2d hidden)
70%|███████   | 7/10 [00:03<00:01,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:09[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.3 ms (8813.56 MiB), compute=136.9 ms, exposed_stall=313.8 ms, wall=432.6 ms, hidden_h2d=136.5 ms (30% of h2d hidden)
80%|████████  | 8/10 [00:03<00:00,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:09[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.2 ms (8813.56 MiB), compute=136.8 ms, exposed_stall=314.8 ms, wall=433.5 ms, hidden_h2d=136.4 ms (30% of h2d hidden)
90%|█████████ | 9/10 [00:04<00:00,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:10[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.7 ms (8813.56 MiB), compute=136.8 ms, exposed_stall=314.3 ms, wall=433.0 ms, hidden_h2d=136.4 ms (30% of h2d hidden)
0%|          | 0/10 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:10[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.1 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=313.9 ms, wall=432.2 ms, hidden_h2d=136.2 ms (30% of h2d hidden)
10%|█         | 1/10 [00:00<00:04,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:11[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.3 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=315.1 ms, wall=433.3 ms, hidden_h2d=136.2 ms (30% of h2d hidden)
20%|██        | 2/10 [00:00<00:03,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:11[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=454.0 ms (8813.56 MiB), compute=136.6 ms, exposed_stall=317.8 ms, wall=435.2 ms, hidden_h2d=136.3 ms (30% of h2d hidden)
30%|███       | 3/10 [00:01<00:03,  2.13it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:12[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.4 ms (8813.56 MiB), compute=136.8 ms, exposed_stall=314.9 ms, wall=433.7 ms, hidden_h2d=136.5 ms (30% of h2d hidden)
40%|████      | 4/10 [00:01<00:02,  2.13it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:12[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.7 ms (8813.56 MiB), compute=136.9 ms, exposed_stall=314.2 ms, wall=433.1 ms, hidden_h2d=136.5 ms (30% of h2d hidden)
50%|█████     | 5/10 [00:02<00:02,  2.13it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:13[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.0 ms (8813.56 MiB), compute=136.8 ms, exposed_stall=314.5 ms, wall=433.3 ms, hidden_h2d=136.5 ms (30% of h2d hidden)
60%|██████    | 6/10 [00:02<00:01,  2.13it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:13[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.3 ms (8813.56 MiB), compute=136.9 ms, exposed_stall=314.9 ms, wall=433.7 ms, hidden_h2d=136.5 ms (30% of h2d hidden)
70%|███████   | 7/10 [00:03<00:01,  2.13it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:13[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.7 ms (8813.56 MiB), compute=136.8 ms, exposed_stall=314.2 ms, wall=433.0 ms, hidden_h2d=136.4 ms (30% of h2d hidden)
80%|████████  | 8/10 [00:03<00:00,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:14[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.3 ms (8813.56 MiB), compute=136.9 ms, exposed_stall=314.8 ms, wall=433.6 ms, hidden_h2d=136.5 ms (30% of h2d hidden)
90%|█████████ | 9/10 [00:04<00:00,  2.13it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:14[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.8 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=314.8 ms, wall=433.2 ms, hidden_h2d=136.0 ms (30% of h2d hidden)
0%|          | 0/10 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:15[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=452.8 ms (8813.56 MiB), compute=137.1 ms, exposed_stall=316.1 ms, wall=435.1 ms, hidden_h2d=136.7 ms (30% of h2d hidden)
10%|█         | 1/10 [00:00<00:04,  2.13it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:15[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.8 ms (8813.56 MiB), compute=136.9 ms, exposed_stall=314.2 ms, wall=433.1 ms, hidden_h2d=136.6 ms (30% of h2d hidden)
20%|██        | 2/10 [00:00<00:03,  2.13it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:16[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.0 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=314.9 ms, wall=433.4 ms, hidden_h2d=136.1 ms (30% of h2d hidden)
30%|███       | 3/10 [00:01<00:03,  2.13it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:16[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.3 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=314.2 ms, wall=432.7 ms, hidden_h2d=136.1 ms (30% of h2d hidden)
40%|████      | 4/10 [00:01<00:02,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:17[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.0 ms (8813.56 MiB), compute=136.8 ms, exposed_stall=314.5 ms, wall=433.2 ms, hidden_h2d=136.5 ms (30% of h2d hidden)
50%|█████     | 5/10 [00:02<00:02,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:17[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.2 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=315.0 ms, wall=433.6 ms, hidden_h2d=136.2 ms (30% of h2d hidden)
60%|██████    | 6/10 [00:02<00:01,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:18[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.7 ms (8813.56 MiB), compute=136.7 ms, exposed_stall=314.3 ms, wall=433.0 ms, hidden_h2d=136.3 ms (30% of h2d hidden)
70%|███████   | 7/10 [00:03<00:01,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:18[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.2 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=314.1 ms, wall=432.5 ms, hidden_h2d=136.1 ms (30% of h2d hidden)
80%|████████  | 8/10 [00:03<00:00,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:19[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=450.3 ms (8813.56 MiB), compute=136.9 ms, exposed_stall=313.9 ms, wall=432.7 ms, hidden_h2d=136.5 ms (30% of h2d hidden)
90%|█████████ | 9/10 [00:04<00:00,  2.14it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:19[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=451.4 ms (8813.56 MiB), compute=136.5 ms, exposed_stall=315.4 ms, wall=433.8 ms, hidden_h2d=136.1 ms (30% of h2d hidden)
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:55:19[0m [90m[layerwise_backend.py:655][0m Layer-wise offloading disabled
```

### fp8_bothflags

```text
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:04[0m [90m[diffusers_loader.py:711][0m Online quantization with CPU offload, using cuda for weight loading (will offload back to CPU)
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:10[0m [90m[diffusers_loader.py:510][0m Stream-offloaded 0 online-quantized layers to CPU during weight loading
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:10[0m [90m[plain_fp8.py:191][0m Unpacked 318 torchao Float8Tensor linear(s) into plain (fp8 weight, fp32 row-scale) parameters
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:15[0m [90m[diffusers_loader.py:887][0m Quantization complete, offloaded model back to CPU
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:15[0m [90m[__init__.py:179][0m Enabling offloader backend: LayerWiseOffloadBackend
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:19[0m [90m[layerwise_backend.py:614][0m Layer-wise offloading enabled on 40 layers (blocks)
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:47[0m [90m[layerwise_backend.py:655][0m Layer-wise offloading disabled
```

### fp8_dlo_n0

```text
[0;36m(DiffusionWorker pid=14732)[0;0m [32mINFO[0m [90m10-09 15:08:25[0m [90m[diffusers_loader.py:711][0m Online quantization with CPU offload, using cuda for weight loading (will offload back to CPU)
[0;36m(DiffusionWorker pid=14732)[0;0m [32mINFO[0m [90m10-09 15:08:27[0m [90m[diffusers_loader.py:857][0m DLO direct checkpoint mmap unavailable; using ordinary loader: no compatible safetensors entries were found in the loader's model sources
[0;36m(DiffusionWorker pid=14732)[0;0m [32mINFO[0m [90m10-09 15:08:31[0m [90m[diffusers_loader.py:510][0m Stream-offloaded 0 online-quantized layers to CPU during weight loading
[0;36m(DiffusionWorker pid=14732)[0;0m [32mINFO[0m [90m10-09 15:08:31[0m [90m[plain_fp8.py:191][0m Unpacked 318 torchao Float8Tensor linear(s) into plain (fp8 weight, fp32 row-scale) parameters
[0;36m(DiffusionWorker pid=14732)[0;0m [32mINFO[0m [90m10-09 15:08:36[0m [90m[diffusers_loader.py:887][0m Quantization complete, offloaded model back to CPU
[0;36m(DiffusionWorker pid=14732)[0;0m [32mINFO[0m [90m10-09 15:08:36[0m [90m[base.py:189][0m Distributed layerwise offload: all selected components use rank-local transfer (no DLO shard or AllGather)
[0;36m(DiffusionWorker pid=14732)[0;0m [32mINFO[0m [90m10-09 15:08:36[0m [90m[__init__.py:179][0m Enabling offloader backend: DistributedLayerwiseOffloadBackend
[0;36m(DiffusionWorker pid=14732)[0;0m [32mINFO[0m [90m10-09 15:08:36[0m [90m[distributed_layerwise_backend.py:1688][0m DLO is using host tensors materialized by the ordinary loader
[0;36m(DiffusionWorker pid=14732)[0;0m [32mINFO[0m [90m10-09 15:08:40[0m [90m[distributed_layerwise_backend.py:1799][0m Distributed layer-wise offloading enabled on 40 blocks across 1 group(s), transfers={dit: rank-local}, unified shared_buffers=2
[0;36m(DiffusionWorker pid=14732)[0;0m [32mINFO[0m [90m10-09 15:09:14[0m [90m[distributed_layerwise_backend.py:1978][0m Distributed layer-wise offloading disabled
```

### fp8_dlo_n4

```text
[0;36m(DiffusionWorker pid=15737)[0;0m [32mINFO[0m [90m10-09 15:09:26[0m [90m[diffusers_loader.py:711][0m Online quantization with CPU offload, using cuda for weight loading (will offload back to CPU)
[0;36m(DiffusionWorker pid=15737)[0;0m [32mINFO[0m [90m10-09 15:09:28[0m [90m[diffusers_loader.py:857][0m DLO direct checkpoint mmap unavailable; using ordinary loader: no compatible safetensors entries were found in the loader's model sources
[0;36m(DiffusionWorker pid=15737)[0;0m [32mINFO[0m [90m10-09 15:09:32[0m [90m[diffusers_loader.py:510][0m Stream-offloaded 0 online-quantized layers to CPU during weight loading
[0;36m(DiffusionWorker pid=15737)[0;0m [32mINFO[0m [90m10-09 15:09:32[0m [90m[plain_fp8.py:191][0m Unpacked 318 torchao Float8Tensor linear(s) into plain (fp8 weight, fp32 row-scale) parameters
[0;36m(DiffusionWorker pid=15737)[0;0m [32mINFO[0m [90m10-09 15:09:36[0m [90m[diffusers_loader.py:887][0m Quantization complete, offloaded model back to CPU
[0;36m(DiffusionWorker pid=15737)[0;0m [32mINFO[0m [90m10-09 15:09:37[0m [90m[base.py:189][0m Distributed layerwise offload: all selected components use rank-local transfer (no DLO shard or AllGather)
[0;36m(DiffusionWorker pid=15737)[0;0m [32mINFO[0m [90m10-09 15:09:37[0m [90m[__init__.py:179][0m Enabling offloader backend: DistributedLayerwiseOffloadBackend
[0;36m(DiffusionWorker pid=15737)[0;0m [32mINFO[0m [90m10-09 15:09:37[0m [90m[distributed_layerwise_backend.py:1688][0m DLO is using host tensors materialized by the ordinary loader
[0;36m(DiffusionWorker pid=15737)[0;0m [32mINFO[0m [90m10-09 15:09:38[0m [90m[distributed_layerwise_backend.py:1710][0m Keeping 4 leading blocks resident on transformer; streaming 36 tail blocks
[0;36m(DiffusionWorker pid=15737)[0;0m [32mINFO[0m [90m10-09 15:09:41[0m [90m[distributed_layerwise_backend.py:1799][0m Distributed layer-wise offloading enabled on 36 blocks across 1 group(s), transfers={dit: rank-local}, unified shared_buffers=2
[0;36m(DiffusionWorker pid=15737)[0;0m [32mINFO[0m [90m10-09 15:10:09[0m [90m[distributed_layerwise_backend.py:1978][0m Distributed layer-wise offloading disabled
```

### bf16_model

```text
[0;36m(DiffusionWorker pid=7049)[0;0m [32mINFO[0m [90m10-09 14:57:00[0m [90m[__init__.py:179][0m Enabling offloader backend: ModelLevelOffloadBackend
[0;36m(DiffusionWorker pid=7049)[0;0m [32mINFO[0m [90m10-09 14:57:01[0m [90m[sequential_backend.py:382][0m Model-level offloading enabled: transformer <-> mllm (mutual exclusion)
[0;36m(DiffusionWorker pid=7049)[0;0m [32mINFO[0m [90m10-09 14:58:10[0m [90m[sequential_backend.py:408][0m Model-level offloading disabled
```

### bf16_layerwise

```text
[0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:44[0m [90m[__init__.py:179][0m Enabling offloader backend: LayerWiseOffloadBackend
[0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:46[0m [90m[layerwise_backend.py:598][0m layerwise offload timing instrumentation enabled (VLLM_OMNI_OFFLOAD_TIMING=1)
[0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:51[0m [90m[layerwise_backend.py:614][0m Layer-wise offloading enabled on 40 layers (blocks)
0%|          | 0/2 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:57[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=911.4 ms (17601.47 MiB), compute=2811.6 ms, exposed_stall=635.2 ms, wall=3409.7 ms, hidden_h2d=276.2 ms (30% of h2d hidden)
50%|█████     | 1/2 [00:04<00:04,  4.49s/it][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:58[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.3 ms (17601.47 MiB), compute=223.6 ms, exposed_stall=676.1 ms, wall=862.3 ms, hidden_h2d=223.2 ms (25% of h2d hidden)
0%|          | 0/10 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:03[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=901.1 ms (17601.47 MiB), compute=1179.3 ms, exposed_stall=735.0 ms, wall=1874.5 ms, hidden_h2d=166.1 ms (18% of h2d hidden)
10%|█         | 1/10 [00:01<00:17,  1.92s/it][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:04[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=900.1 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=789.3 ms, wall=860.8 ms, hidden_h2d=110.8 ms (12% of h2d hidden)
20%|██        | 2/10 [00:02<00:10,  1.33s/it][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:04[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=900.7 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=789.7 ms, wall=861.1 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
30%|███       | 3/10 [00:03<00:07,  1.14s/it][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:05[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=901.0 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=790.1 ms, wall=861.3 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
40%|████      | 4/10 [00:04<00:06,  1.05s/it][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:06[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.5 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=788.7 ms, wall=860.1 ms, hidden_h2d=110.7 ms (12% of h2d hidden)
50%|█████     | 5/10 [00:05<00:04,  1.00it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:07[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=900.3 ms (17601.47 MiB), compute=111.6 ms, exposed_stall=789.1 ms, wall=860.9 ms, hidden_h2d=111.2 ms (12% of h2d hidden)
60%|██████    | 6/10 [00:06<00:03,  1.03it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:08[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.5 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=788.6 ms, wall=860.0 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
70%|███████   | 7/10 [00:07<00:02,  1.05it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:09[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=900.3 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=789.4 ms, wall=860.7 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
80%|████████  | 8/10 [00:08<00:01,  1.07it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:10[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=900.1 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=789.3 ms, wall=860.8 ms, hidden_h2d=110.8 ms (12% of h2d hidden)
90%|█████████ | 9/10 [00:09<00:00,  1.08it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:11[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.8 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=789.0 ms, wall=860.4 ms, hidden_h2d=110.8 ms (12% of h2d hidden)
0%|          | 0/10 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:12[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=900.0 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=789.3 ms, wall=860.8 ms, hidden_h2d=110.7 ms (12% of h2d hidden)
10%|█         | 1/10 [00:00<00:08,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:13[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=900.0 ms (17601.47 MiB), compute=111.2 ms, exposed_stall=789.2 ms, wall=860.6 ms, hidden_h2d=110.8 ms (12% of h2d hidden)
20%|██        | 2/10 [00:01<00:07,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:14[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=901.6 ms (17601.47 MiB), compute=111.6 ms, exposed_stall=790.4 ms, wall=861.7 ms, hidden_h2d=111.2 ms (12% of h2d hidden)
30%|███       | 3/10 [00:02<00:06,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:15[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.0 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=788.2 ms, wall=859.6 ms, hidden_h2d=110.8 ms (12% of h2d hidden)
40%|████      | 4/10 [00:03<00:05,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:15[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.8 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=789.0 ms, wall=860.4 ms, hidden_h2d=110.8 ms (12% of h2d hidden)
50%|█████     | 5/10 [00:04<00:04,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:16[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.5 ms (17601.47 MiB), compute=111.6 ms, exposed_stall=788.3 ms, wall=860.1 ms, hidden_h2d=111.2 ms (12% of h2d hidden)
60%|██████    | 6/10 [00:05<00:03,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:17[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.8 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=789.0 ms, wall=860.1 ms, hidden_h2d=110.8 ms (12% of h2d hidden)
70%|███████   | 7/10 [00:06<00:02,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:18[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.2 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=788.3 ms, wall=859.7 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
80%|████████  | 8/10 [00:07<00:01,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:19[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=900.1 ms (17601.47 MiB), compute=111.6 ms, exposed_stall=788.8 ms, wall=860.6 ms, hidden_h2d=111.3 ms (12% of h2d hidden)
90%|█████████ | 9/10 [00:08<00:00,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:20[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=903.7 ms (17601.47 MiB), compute=111.8 ms, exposed_stall=792.7 ms, wall=864.7 ms, hidden_h2d=111.0 ms (12% of h2d hidden)
0%|          | 0/10 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:21[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=901.2 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=790.3 ms, wall=861.7 ms, hidden_h2d=111.0 ms (12% of h2d hidden)
10%|█         | 1/10 [00:00<00:08,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:22[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=900.6 ms (17601.47 MiB), compute=111.7 ms, exposed_stall=789.3 ms, wall=860.7 ms, hidden_h2d=111.3 ms (12% of h2d hidden)
20%|██        | 2/10 [00:01<00:07,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:23[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.2 ms (17601.47 MiB), compute=111.7 ms, exposed_stall=787.9 ms, wall=859.8 ms, hidden_h2d=111.3 ms (12% of h2d hidden)
30%|███       | 3/10 [00:02<00:06,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:24[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=900.0 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=789.1 ms, wall=860.6 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
40%|████      | 4/10 [00:03<00:05,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:25[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.1 ms (17601.47 MiB), compute=111.2 ms, exposed_stall=788.2 ms, wall=859.6 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
50%|█████     | 5/10 [00:04<00:04,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:26[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.8 ms (17601.47 MiB), compute=111.7 ms, exposed_stall=788.4 ms, wall=860.0 ms, hidden_h2d=111.4 ms (12% of h2d hidden)
60%|██████    | 6/10 [00:05<00:03,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:26[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=898.7 ms (17601.47 MiB), compute=111.2 ms, exposed_stall=787.9 ms, wall=859.3 ms, hidden_h2d=110.8 ms (12% of h2d hidden)
70%|███████   | 7/10 [00:06<00:02,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:27[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.6 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=788.7 ms, wall=860.2 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
80%|████████  | 8/10 [00:07<00:01,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:28[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.7 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=788.8 ms, wall=860.3 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
90%|█████████ | 9/10 [00:08<00:00,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:29[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=900.3 ms (17601.47 MiB), compute=111.7 ms, exposed_stall=789.1 ms, wall=861.2 ms, hidden_h2d=111.2 ms (12% of h2d hidden)
0%|          | 0/10 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:30[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.1 ms (17601.47 MiB), compute=111.7 ms, exposed_stall=787.8 ms, wall=859.3 ms, hidden_h2d=111.3 ms (12% of h2d hidden)
10%|█         | 1/10 [00:00<00:08,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:31[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.7 ms (17601.47 MiB), compute=111.2 ms, exposed_stall=788.8 ms, wall=860.2 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
20%|██        | 2/10 [00:01<00:07,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:32[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.2 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=788.3 ms, wall=859.7 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
30%|███       | 3/10 [00:02<00:06,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:33[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.7 ms (17601.47 MiB), compute=111.7 ms, exposed_stall=788.4 ms, wall=859.6 ms, hidden_h2d=111.3 ms (12% of h2d hidden)
40%|████      | 4/10 [00:03<00:05,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:34[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.2 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=788.4 ms, wall=859.6 ms, hidden_h2d=110.8 ms (12% of h2d hidden)
50%|█████     | 5/10 [00:04<00:04,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:35[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=898.6 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=787.7 ms, wall=859.2 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
60%|██████    | 6/10 [00:05<00:03,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:36[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.6 ms (17601.47 MiB), compute=111.4 ms, exposed_stall=788.6 ms, wall=860.0 ms, hidden_h2d=111.0 ms (12% of h2d hidden)
70%|███████   | 7/10 [00:06<00:02,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:37[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.1 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=788.1 ms, wall=859.5 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
80%|████████  | 8/10 [00:07<00:01,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:37[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=900.1 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=789.1 ms, wall=860.5 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
90%|█████████ | 9/10 [00:08<00:00,  1.10it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:38[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=898.6 ms (17601.47 MiB), compute=111.3 ms, exposed_stall=787.6 ms, wall=859.1 ms, hidden_h2d=110.9 ms (12% of h2d hidden)
[0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:59:38[0m [90m[layerwise_backend.py:655][0m Layer-wise offloading disabled
```

### dlo_n0

```text
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 14:59:53[0m [90m[diffusers_loader.py:857][0m DLO direct checkpoint mmap unavailable; using ordinary loader: 186 required DiT tensors have no checkpoint binding (first 5: ['transformer.context_refiner.0.attn.to_out.weight', 'transformer.context_refiner.0.attn.to_qkv.weight', 'transformer.context_refiner.0.feed_forward.gate_up_proj.weight', 'transformer.context_refiner.1.attn.to_out.weight', 'transformer.context_refiner.1.attn.to_qkv.weight'])
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 14:59:54[0m [90m[base.py:189][0m Distributed layerwise offload: all selected components use rank-local transfer (no DLO shard or AllGather)
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 14:59:54[0m [90m[__init__.py:179][0m Enabling offloader backend: DistributedLayerwiseOffloadBackend
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 14:59:54[0m [90m[distributed_layerwise_backend.py:1688][0m DLO is using host tensors materialized by the ordinary loader
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 15:00:00[0m [90m[distributed_layerwise_backend.py:1799][0m Distributed layer-wise offloading enabled on 40 blocks across 1 group(s), transfers={dit: rank-local}, unified shared_buffers=2
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 15:00:52[0m [90m[distributed_layerwise_backend.py:1978][0m Distributed layer-wise offloading disabled
```

### dlo_n4

```text
(no matching lines)
```

### dlo_n4_retry1

```text
[0;36m(DiffusionWorker pid=12596)[0;0m [32mINFO[0m [90m10-09 15:03:24[0m [90m[diffusers_loader.py:857][0m DLO direct checkpoint mmap unavailable; using ordinary loader: 186 required DiT tensors have no checkpoint binding (first 5: ['transformer.context_refiner.0.attn.to_out.weight', 'transformer.context_refiner.0.attn.to_qkv.weight', 'transformer.context_refiner.0.feed_forward.gate_up_proj.weight', 'transformer.context_refiner.1.attn.to_out.weight', 'transformer.context_refiner.1.attn.to_qkv.weight'])
[0;36m(DiffusionWorker pid=12596)[0;0m [32mINFO[0m [90m10-09 15:03:26[0m [90m[base.py:189][0m Distributed layerwise offload: all selected components use rank-local transfer (no DLO shard or AllGather)
[0;36m(DiffusionWorker pid=12596)[0;0m [32mINFO[0m [90m10-09 15:03:26[0m [90m[__init__.py:179][0m Enabling offloader backend: DistributedLayerwiseOffloadBackend
[0;36m(DiffusionWorker pid=12596)[0;0m [32mINFO[0m [90m10-09 15:03:26[0m [90m[distributed_layerwise_backend.py:1688][0m DLO is using host tensors materialized by the ordinary loader
[0;36m(DiffusionWorker pid=12596)[0;0m [32mINFO[0m [90m10-09 15:03:26[0m [90m[distributed_layerwise_backend.py:1710][0m Keeping 4 leading blocks resident on transformer; streaming 36 tail blocks
[0;36m(DiffusionWorker pid=12596)[0;0m [32mINFO[0m [90m10-09 15:03:32[0m [90m[distributed_layerwise_backend.py:1799][0m Distributed layer-wise offloading enabled on 36 blocks across 1 group(s), transfers={dit: rank-local}, unified shared_buffers=2
[0;36m(DiffusionWorker pid=12596)[0;0m [32mINFO[0m [90m10-09 15:04:17[0m [90m[distributed_layerwise_backend.py:1978][0m Distributed layer-wise offloading disabled
```

### dlo_n4_retry2

```text
[0;36m(DiffusionWorker pid=13641)[0;0m [32mINFO[0m [90m10-09 15:04:30[0m [90m[diffusers_loader.py:857][0m DLO direct checkpoint mmap unavailable; using ordinary loader: 186 required DiT tensors have no checkpoint binding (first 5: ['transformer.context_refiner.0.attn.to_out.weight', 'transformer.context_refiner.0.attn.to_qkv.weight', 'transformer.context_refiner.0.feed_forward.gate_up_proj.weight', 'transformer.context_refiner.1.attn.to_out.weight', 'transformer.context_refiner.1.attn.to_qkv.weight'])
[0;36m(DiffusionWorker pid=13641)[0;0m [32mINFO[0m [90m10-09 15:04:32[0m [90m[base.py:189][0m Distributed layerwise offload: all selected components use rank-local transfer (no DLO shard or AllGather)
[0;36m(DiffusionWorker pid=13641)[0;0m [32mINFO[0m [90m10-09 15:04:32[0m [90m[__init__.py:179][0m Enabling offloader backend: DistributedLayerwiseOffloadBackend
[0;36m(DiffusionWorker pid=13641)[0;0m [32mINFO[0m [90m10-09 15:04:32[0m [90m[distributed_layerwise_backend.py:1688][0m DLO is using host tensors materialized by the ordinary loader
[0;36m(DiffusionWorker pid=13641)[0;0m [32mINFO[0m [90m10-09 15:04:32[0m [90m[distributed_layerwise_backend.py:1710][0m Keeping 4 leading blocks resident on transformer; streaming 36 tail blocks
[0;36m(DiffusionWorker pid=13641)[0;0m [32mINFO[0m [90m10-09 15:04:38[0m [90m[distributed_layerwise_backend.py:1799][0m Distributed layer-wise offloading enabled on 36 blocks across 1 group(s), transfers={dit: rank-local}, unified shared_buffers=2
[0;36m(DiffusionWorker pid=13641)[0;0m [32mINFO[0m [90m10-09 15:05:23[0m [90m[distributed_layerwise_backend.py:1978][0m Distributed layer-wise offloading disabled
```

