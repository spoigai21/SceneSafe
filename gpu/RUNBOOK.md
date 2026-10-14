# NVIDIA session runbook (Phase 6, lane `gpu-nvidia-a10`)

**Goal:** one rented **Lambda Cloud 1x A10** instance runs a script that was already tested on the laptop, and the
results come back. There is no debugging on rented time: if the smoke test fails, copy the log back and **terminate**.

**Expected:** about **45–60 minutes** of instance time, which is **about $1.00–1.30** at $1.29 per hour (on-demand,
plus sales tax, price checked 2026-10-13). Hard stop at 2 hours ($2.60).

| Step | Where | Time |
|---|---|---|
| 0. Build the bundle | laptop | 1 min |
| 1. Launch the instance | Lambda dashboard | 2–5 min to boot (billing starts here) |
| 2. Upload and unpack | laptop → instance | under 1 min (bundle is about 6 MB) |
| 3. `setup.sh` | instance | 8–12 min (packages, about 10 GB of models, 1.5 GB of films from the Internet Archive) |
| 4. `smoke` | instance | 4–5 min (vLLM start-up is most of it) |
| 5. `time` | instance | 25–35 min |
| 6. Copy the results back, then TERMINATE | laptop + dashboard | 2 min |

## 0. Laptop: build the bundle

```bash
cd ~/Desktop/scene-safe
bash gpu/bundle.sh                 # writes ~/gpu_bundle.tar
```

If the Internet Archive is slow on the day, use `bash gpu/bundle.sh --with-films` instead. That ships the 3 films, so
the bundle is about 1.5 GB and the upload takes longer from home, but setup skips the download.

## 1. Lambda: launch

1. Go to cloud.lambda.ai → **Instances → Launch instance**.
2. Choose **1x A10 (24 GB PCIe)**, any region with capacity, and the default image (Ubuntu + Lambda Stack).
3. Attach your SSH key. No filesystem is needed.
4. Wait for **Running** and copy its IP address.

## 2. Upload and unpack

```bash
IP=<instance ip>
scp ~/gpu_bundle.tar ubuntu@$IP:~
ssh ubuntu@$IP
mkdir -p ss && tar -xf gpu_bundle.tar -C ss && cd ss
tmux new -s run              # so a dropped connection does not stop the run; reattach with: tmux attach -t run
```

## 3–5. Set up, smoke test, time

```bash
bash gpu/setup.sh 2>&1 | tee setup.log                        # ends with "SETUP OK"
source gpu/env.sh
python gpu/run_gpu.py smoke 2>&1 | tee smoke.log              # ends with "SMOKE OK"
python gpu/run_gpu.py time 2>&1 | tee time.log                # ends with "wrote results/phase6/gpu_gpu-nvidia-a10_seed0.json"
```

**If `setup.sh` or `smoke` fails:** stop there. Copy the logs back (step 6), terminate the instance, and fix the problem
on the laptop.

**Failures that do not stop the run:**
- **NVDEC:** if this PyAV build has no CUDA decoder, the smoke test prints `nvdec unavailable` and carries on.
  `pyav_hw_devices` in the results shows what was available.
- **A stage that fails during `time`:** it is recorded under `errors` in the results JSON and the other stages
  continue. The cost step then marks that NVIDIA total "incomplete" instead of estimating it.

## 6. Copy back, then TERMINATE

From the laptop:

```bash
scp ubuntu@$IP:~/ss/results/phase6/gpu_gpu-nvidia-a10_seed0.json ~/Desktop/scene-safe/results/phase6/
mkdir -p ~/Desktop/scene-safe/results/phase6/gpu_logs && scp ubuntu@$IP:"~/ss/*.log" ~/Desktop/scene-safe/results/phase6/gpu_logs/
```

Then in the Lambda dashboard select the instance and click **Terminate**. On Lambda there is no "stop": a running
instance bills until it is terminated. Check that the instance list is empty.

Back on the laptop:

```bash
PYTHONPATH=. .venv/bin/python scripts/run_phase6.py cost      # adds the NVIDIA lane; the laptop figures are unchanged
```

## What the run measures, and how it differs from the laptop

**Same as the laptop run** (`scripts/run_phase6.py`, seed 0):
- the same 3 films and 10-minute Whisper windows (20:00–30:00)
- the same 60 scenes, with the same strip and grid images
- the same prompts, imported from `scenesafe/`
- the same token limits and greedy decoding
- 2 strip checks per scene
- all 4,922 card texts embedded
- 3 runs of each stage, median and range
- model loading and one warm-up call excluded

**Different, and recorded in the results:**
- **VLM:** Qwen3-VL-4B-Instruct in **bf16 on vLLM**. The laptop runs it in **MLX 4-bit**. Prefix and image caching are
  turned off, so the repeat runs do real work.
- **Whisper:** large-v3-turbo on **faster-whisper** (float16, beam 1). The laptop uses mlx-whisper.
- **Two VLM modes:**
  - **sequential:** one request at a time, the same pattern as the laptop
  - **batched:** all requests in one call, the normal way to use vLLM
- **Image creation:** timed on 60 scenes of the 3 shipped films, because only those films are on the machine. The laptop
  timed it on its 60 scenes from all 19 films.

So the comparison is **hardware plus software**, not hardware alone.

## Checked on the laptop, and not checked

**Checked:**
- `gpu/bundle.sh` builds the bundle, and the unpacked bundle runs `smoke --dry` and `time --dry` on its own: CPU, 1–2
  items, Whisper tiny, and a stand-in for vLLM.
- These code paths ran for real:
  - CPU decode + CLIP
  - audio decode and faster-whisper
  - strip and grid images
  - the text-embedding child process
  - results writing and the cost maths
- The chat prompt built here is **character-for-character identical** to the laptop's mlx-vlm prompt, checked for the
  card prompt with the real Qwen3-VL processor.
- The yes/no token ids match the laptop's `verify_p`.
- The P(yes) maths was tested on hand-made inputs.
- The vLLM calls were checked against the vLLM 0.31.0 source:
  - `LLM(...)` arguments: `limit_mm_per_prompt`, `enable_prefix_caching`, `mm_processor_cache_gb`
  - `SamplingParams.logprob_token_ids`, which returns exact log-probabilities for yes/Yes/no/No
  - both logprob output formats
  - Qwen3-VL is in the model registry

**Not checked (no NVIDIA GPU here), so `smoke` exercises each one first:**
- vLLM actually loading Qwen3-VL on the A10.
- faster-whisper finding CUDA 12 cuBLAS and cuDNN 9. `setup.sh` installs them and `env.sh` adds them to
  `LD_LIBRARY_PATH`.
- Whether PyAV's Linux wheel has NVDEC.
- `--torch-backend=auto` choosing a torch build that this instance's driver supports.
