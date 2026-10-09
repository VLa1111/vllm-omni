| Arm | Checkpoint | Offload mode | Role | Peak MiB (steady) | Peak MiB (load) | E2E s min-max (n=3) | PNG sha256 | Status |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| bf16_noffload | bf16 (34.6 GiB) | none | expected rejection | - | 18352 | - | - | OOM (expected rejection) |
| bf16_model | bf16 (34.6 GiB) | module-level | A only | 21924 | 21850 | 5.65-5.67 | de4c4747064d5537 | ok |
| bf16_layerwise | bf16 (34.6 GiB) | local layerwise | A only | 21380 | 21368 | 9.17-9.18 | de4c4747064d5537 | ok |
| dlo_n0 | bf16 (34.6 GiB) | distributed layerwise N=0 | A only | 21038 | 21026 | 9.15-9.16 | de4c4747064d5537 | ok |
| dlo_n4 | bf16 (34.6 GiB) | distributed layerwise N=4 | A only | - | 0 | - | - | no measured requests |
| fp8_noffload | Base-fp8 | none | B only | 21132 | 21284 | 1.60-1.61 | 554b4412506613e5 | ok |
| fp8_model | Base-fp8 | module-level | A+B | 11786 | 21100 | 3.96-3.97 | 7752d736ca3b4567 | ok |
| fp8_layerwise | Base-fp8 | local layerwise | A+B | 13194 | 21100 | 4.78-4.83 | 7752d736ca3b4567 | ok |
| fp8_bothflags | Base-fp8 | module+layerwise both flags | mode-boundary check | 13194 | 21100 | 4.76-4.77 | 7752d736ca3b4567 | ok |

### bf16_noffload

```json
{
  "arm": "bf16_noffload"
}
```

### bf16_model

```json
{
  "arm": "bf16_model",
  "response_keys": [
    "cot_output",
    "created",
    "data",
    "metrics",
    "output_format",
    "size"
  ],
  "data0_keys": [
    "b64_json",
    "revised_prompt",
    "url"
  ],
  "req_ms": [
    5648,
    5666,
    5654
  ],
  "req_median_ms": 5654
}
```
- req0: 18069 ms, png de4c4747064d5537
- req1: 5648 ms, png de4c4747064d5537
- req2: 5666 ms, png de4c4747064d5537
- req3: 5654 ms, png de4c4747064d5537

Activation evidence:

```text
[0;36m(DiffusionWorker pid=7049)[0;0m [32mINFO[0m [90m10-09 14:57:00[0m [90m[__init__.py:179][0m Enabling offloader backend: ModelLevelOffloadBackend
[0;36m(DiffusionWorker pid=7049)[0;0m [32mINFO[0m [90m10-09 14:57:01[0m [90m[sequential_backend.py:382][0m Model-level offloading enabled: transformer <-> mllm (mutual exclusion)
[0;36m(DiffusionWorker pid=7049)[0;0m [32mINFO[0m [90m10-09 14:57:31[0m [90m[launcher.py:80][0m Route: /v1/audio/speech/stream, Endpoint: streaming_speech
[0;36m(DiffusionWorker pid=7049)[0;0m [32mINFO[0m [90m10-09 14:57:31[0m [90m[launcher.py:80][0m Route: /v1/video/chat/stream, Endpoint: streaming_video_chat
[0;36m(DiffusionWorker pid=7049)[0;0m [32mINFO[0m [90m10-09 14:57:31[0m [90m[launcher.py:80][0m Route: /v1/realtime/video, Endpoint: streaming_video_output
```

### bf16_layerwise

```json
{
  "arm": "bf16_layerwise",
  "response_keys": [
    "cot_output",
    "created",
    "data",
    "metrics",
    "output_format",
    "size"
  ],
  "data0_keys": [
    "b64_json",
    "revised_prompt",
    "url"
  ],
  "req_ms": [
    9170,
    9175,
    9167
  ],
  "req_median_ms": 9170
}
```
- req0: 10229 ms, png de4c4747064d5537
- req1: 9170 ms, png de4c4747064d5537
- req2: 9175 ms, png de4c4747064d5537
- req3: 9167 ms, png de4c4747064d5537

Activation evidence:

```text
[0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:44[0m [90m[__init__.py:179][0m Enabling offloader backend: LayerWiseOffloadBackend
[0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:45[0m [90m[layerwise_backend.py:578][0m Applying layer-wise offloading on ['transformer']
[0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:46[0m [90m[layerwise_backend.py:598][0m layerwise offload timing instrumentation enabled (VLLM_OMNI_OFFLOAD_TIMING=1)
[0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:51[0m [90m[layerwise_backend.py:614][0m Layer-wise offloading enabled on 40 layers (blocks)

  0%|          | 0/2 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:57[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=911.4 ms (17601.47 MiB), compute=2811.6 ms, exposed_stall=635.2 ms, wall=3409.7 ms, hidden_h2d=276.2 ms (30% of h2d hidden)

 50%|█████     | 1/2 [00:04<00:04,  4.49s/it][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:58[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=899.3 ms (17601.47 MiB), compute=223.6 ms, exposed_stall=676.1 ms, wall=862.3 ms, hidden_h2d=223.2 ms (25% of h2d hidden)
[0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:58[0m [90m[launcher.py:80][0m Route: /v1/audio/speech/stream, Endpoint: streaming_speech
[0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:58[0m [90m[launcher.py:80][0m Route: /v1/video/chat/stream, Endpoint: streaming_video_chat
[0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:46[0m [90m[layerwise_backend.py:598][0m layerwise offload timing instrumentation enabled (VLLM_OMNI_OFFLOAD_TIMING=1)

  0%|          | 0/2 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=8616)[0;0m [32mINFO[0m [90m10-09 14:58:57[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=911.4 ms (17601.47 MiB), compute=2811.6 ms, exposed_stall=635.2 ms, wall=3409.7 ms, hidden_h2d=276.2 ms (30% of h2d hidden)
```

### dlo_n0

```json
{
  "arm": "dlo_n0",
  "response_keys": [
    "cot_output",
    "created",
    "data",
    "metrics",
    "output_format",
    "size"
  ],
  "data0_keys": [
    "b64_json",
    "revised_prompt",
    "url"
  ],
  "req_ms": [
    9148,
    9161,
    9162
  ],
  "req_median_ms": 9161
}
```
- req0: 10234 ms, png de4c4747064d5537
- req1: 9148 ms, png de4c4747064d5537
- req2: 9161 ms, png de4c4747064d5537
- req3: 9162 ms, png de4c4747064d5537

Activation evidence:

```text
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 14:59:54[0m [90m[base.py:189][0m Distributed layerwise offload: all selected components use rank-local transfer (no DLO shard or AllGather)
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 14:59:54[0m [90m[__init__.py:179][0m Enabling offloader backend: DistributedLayerwiseOffloadBackend
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 14:59:54[0m [90m[distributed_layerwise_backend.py:1700][0m Applying distributed layer-wise offloading on ['transformer']
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 15:00:00[0m [90m[distributed_layerwise_backend.py:1799][0m Distributed layer-wise offloading enabled on 40 blocks across 1 group(s), transfers={dit: rank-local}, unified shared_buffers=2
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 15:00:11[0m [90m[launcher.py:80][0m Route: /v1/audio/speech/stream, Endpoint: streaming_speech
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 15:00:11[0m [90m[launcher.py:80][0m Route: /v1/video/chat/stream, Endpoint: streaming_video_chat
[0;36m(DiffusionWorker pid=9785)[0;0m [32mINFO[0m [90m10-09 15:00:11[0m [90m[launcher.py:80][0m Route: /v1/realtime/video, Endpoint: streaming_video_output
```

### dlo_n4

```json
{
  "arm": "dlo_n4"
}
```

### fp8_noffload

```json
{
  "arm": "fp8_noffload",
  "response_keys": [
    "cot_output",
    "created",
    "data",
    "metrics",
    "output_format",
    "size"
  ],
  "data0_keys": [
    "b64_json",
    "revised_prompt",
    "url"
  ],
  "req_ms": [
    1597,
    1614,
    1596
  ],
  "req_median_ms": 1597
}
```
- req0: 9550 ms, png 554b4412506613e5
- req1: 1597 ms, png 554b4412506613e5
- req2: 1614 ms, png 554b4412506613e5
- req3: 1596 ms, png 554b4412506613e5

Activation evidence:

```text
[0;36m(DiffusionWorker pid=2867)[0;0m [32mINFO[0m [90m10-09 14:53:12[0m [90m[launcher.py:80][0m Route: /v1/audio/speech/stream, Endpoint: streaming_speech
[0;36m(DiffusionWorker pid=2867)[0;0m [32mINFO[0m [90m10-09 14:53:12[0m [90m[launcher.py:80][0m Route: /v1/video/chat/stream, Endpoint: streaming_video_chat
[0;36m(DiffusionWorker pid=2867)[0;0m [32mINFO[0m [90m10-09 14:53:12[0m [90m[launcher.py:80][0m Route: /v1/realtime/video, Endpoint: streaming_video_output
```

### fp8_model

```json
{
  "arm": "fp8_model",
  "response_keys": [
    "cot_output",
    "created",
    "data",
    "metrics",
    "output_format",
    "size"
  ],
  "data0_keys": [
    "b64_json",
    "revised_prompt",
    "url"
  ],
  "req_ms": [
    3959,
    3965,
    3974
  ],
  "req_median_ms": 3965
}
```
- req0: 7307 ms, png 7752d736ca3b4567
- req1: 3959 ms, png 7752d736ca3b4567
- req2: 3965 ms, png 7752d736ca3b4567
- req3: 3974 ms, png 7752d736ca3b4567

Activation evidence:

```text
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:53:41[0m [90m[diffusers_loader.py:711][0m Online quantization with CPU offload, using cuda for weight loading (will offload back to CPU)
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:53:47[0m [90m[diffusers_loader.py:510][0m Stream-offloaded 0 online-quantized layers to CPU during weight loading
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:53:52[0m [90m[diffusers_loader.py:887][0m Quantization complete, offloaded model back to CPU
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:53:53[0m [90m[__init__.py:179][0m Enabling offloader backend: ModelLevelOffloadBackend
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:53:53[0m [90m[sequential_backend.py:382][0m Model-level offloading enabled: transformer <-> mllm (mutual exclusion)
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:54:02[0m [90m[launcher.py:80][0m Route: /v1/audio/speech/stream, Endpoint: streaming_speech
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:54:02[0m [90m[launcher.py:80][0m Route: /v1/video/chat/stream, Endpoint: streaming_video_chat
[0;36m(DiffusionWorker pid=4291)[0;0m [32mINFO[0m [90m10-09 14:54:02[0m [90m[launcher.py:80][0m Route: /v1/realtime/video, Endpoint: streaming_video_output
```

### fp8_layerwise

```json
{
  "arm": "fp8_layerwise",
  "response_keys": [
    "cot_output",
    "created",
    "data",
    "metrics",
    "output_format",
    "size"
  ],
  "data0_keys": [
    "b64_json",
    "revised_prompt",
    "url"
  ],
  "req_ms": [
    4801,
    4780,
    4825
  ],
  "req_median_ms": 4801
}
```
- req0: 6008 ms, png 7752d736ca3b4567
- req1: 4801 ms, png 7752d736ca3b4567
- req2: 4780 ms, png 7752d736ca3b4567
- req3: 4825 ms, png 7752d736ca3b4567

Activation evidence:

```text
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:36[0m [90m[diffusers_loader.py:711][0m Online quantization with CPU offload, using cuda for weight loading (will offload back to CPU)
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:42[0m [90m[diffusers_loader.py:510][0m Stream-offloaded 0 online-quantized layers to CPU during weight loading
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:46[0m [90m[diffusers_loader.py:887][0m Quantization complete, offloaded model back to CPU
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:47[0m [90m[__init__.py:179][0m Enabling offloader backend: LayerWiseOffloadBackend
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:48[0m [90m[layerwise_backend.py:578][0m Applying layer-wise offloading on ['transformer']
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:48[0m [90m[layerwise_backend.py:598][0m layerwise offload timing instrumentation enabled (VLLM_OMNI_OFFLOAD_TIMING=1)
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:50[0m [90m[layerwise_backend.py:614][0m Layer-wise offloading enabled on 40 layers (blocks)

  0%|          | 0/2 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:56[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=452.4 ms (8813.56 MiB), compute=3136.3 ms, exposed_stall=182.7 ms, wall=3303.6 ms, hidden_h2d=269.7 ms (60% of h2d hidden)
[0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:48[0m [90m[layerwise_backend.py:598][0m layerwise offload timing instrumentation enabled (VLLM_OMNI_OFFLOAD_TIMING=1)

  0%|          | 0/2 [00:00<?, ?it/s][0;36m(DiffusionWorker pid=5205)[0;0m [32mINFO[0m [90m10-09 14:54:56[0m [90m[layerwise_backend.py:347][0m layerwise offload timing (round of 40 blocks): h2d=452.4 ms (8813.56 MiB), compute=3136.3 ms, exposed_stall=182.7 ms, wall=3303.6 ms, hidden_h2d=269.7 ms (60% of h2d hidden)
```

### fp8_bothflags

```json
{
  "arm": "fp8_bothflags",
  "response_keys": [
    "cot_output",
    "created",
    "data",
    "metrics",
    "output_format",
    "size"
  ],
  "data0_keys": [
    "b64_json",
    "revised_prompt",
    "url"
  ],
  "req_ms": [
    4758,
    4759,
    4773
  ],
  "req_median_ms": 4759
}
```
- req0: 6068 ms, png 7752d736ca3b4567
- req1: 4758 ms, png 7752d736ca3b4567
- req2: 4759 ms, png 7752d736ca3b4567
- req3: 4773 ms, png 7752d736ca3b4567

Activation evidence:

```text
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:04[0m [90m[diffusers_loader.py:711][0m Online quantization with CPU offload, using cuda for weight loading (will offload back to CPU)
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:10[0m [90m[diffusers_loader.py:510][0m Stream-offloaded 0 online-quantized layers to CPU during weight loading
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:15[0m [90m[diffusers_loader.py:887][0m Quantization complete, offloaded model back to CPU
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:15[0m [90m[__init__.py:179][0m Enabling offloader backend: LayerWiseOffloadBackend
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:16[0m [90m[layerwise_backend.py:578][0m Applying layer-wise offloading on ['transformer']
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:19[0m [90m[layerwise_backend.py:614][0m Layer-wise offloading enabled on 40 layers (blocks)
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:25[0m [90m[launcher.py:80][0m Route: /v1/audio/speech/stream, Endpoint: streaming_speech
[0;36m(DiffusionWorker pid=11537)[0;0m [32mINFO[0m [90m10-09 15:02:25[0m [90m[launcher.py:80][0m Route: /v1/video/chat/stream, Endpoint: streaming_video_chat
```

