# smolDiffusion

**Quantized Stable Diffusion that fits in a shoebox.**

Fine-tuned on a custom meme/character dataset and compressed to run on GPUs with as little as 2–4 GB of VRAM. Built from scratch — no high-level wrappers around the hard parts.

---

## What this is

Standard SD v1.5 is ~4 GB in fp16. Getting it to run on consumer or free-tier hardware requires real engineering, not just loading a smaller model. This project chains together three things:

1. **LoRA fine-tuning** on a custom dataset to specialize the model
2. **Custom Q4_K quantization** of the UNet (860M params → ~450 MB) with per-block int8 scales
3. **On-the-fly dequantization** via custom `QuantizedLinear` and `QuantizedConv2d` layers, so weights decompress only during their forward pass and are immediately freed — no VRAM spike from holding the full fp16 model

The VAE lives on CPU and moves to GPU only for the final decode step. The text encoder does the same. At peak, only the UNet (quantized) and the active layer's dequantized weights occupy GPU memory.

---

## Architecture

```
Prompt
  │
  ▼
CLIP Text Encoder (Q8, CPU-offloaded)
  │  [77 × 768 embeddings]
  ▼
UNet (Q4_K, ~450 MB on GPU)  ◄──── Noisy latent (64×64×4)
  │  runs N denoising steps         Timestep embedding
  │  with classifier-free guidance
  ▼
Clean latent (64×64×4)
  │
  ▼
VAE Decoder (fp16, tiled, CPU-offloaded)
  │
  ▼
Image (up to 512×512)
```

---

## Fine-tuning: LoRA

LoRA decomposes each weight update into two low-rank matrices: `ΔW = B·A` where `B ∈ R^(d×r)` and `A ∈ R^(r×k)`, with rank `r=16`. Applied to the attention projection layers (`to_q`, `to_k`, `to_v`, `to_out`) across all UNet attention blocks. Trainable parameters: ~3–4M out of 860M (~0.4%).

Trained with:
- Batch size 1, gradient accumulation 4 (effective batch 4)
- AdamW, lr=1e-4, weight decay=0.01
- 15 epochs, DDPMScheduler, MSE loss on noise prediction
- Images preprocessed to 512×512 center-crop, normalized to [-1, 1]

After training, LoRA weights are merged into the base model via `merge_and_unload()` — `W_new = W_0 + BA` — producing a clean checkpoint with no PEFT overhead.

---

## Quantization

### UNet: Q4_K

Custom block quantization. For each weight tensor:

1. Flatten and split into blocks of 64
2. Compute per-block scale: `scale = max(|block|) / 7.5`
3. Quantize: `q = round(w / scale).clamp(-7, 7)` → stored as `int8`
4. Store scales separately as `fp16`

Numerically sensitive layers (time embeddings, `conv_out`, mid-block attention output) are kept in fp16 and excluded from quantization. Tensors with fewer than 1024 elements are also excluded.

Result: UNet goes from ~1.7 GB → ~450 MB (3.5–4× compression).

### CLIP: Q8

Simpler global quantization per weight matrix: `scale = max(|W|) / 127`, quantize to `int8` range `[-128, 127]`. CLIP is dequantized fully at load time (not on-the-fly) since it runs once per prompt and the memory is freed immediately after.

---

## On-the-Fly Dequantization

The UNet's linear and conv layers are replaced with custom modules:

```python
class QuantizedLinear(nn.Module):
    def forward(self, x):
        # int8 weights × fp16 scales → fp16 weight matrix, used once, then freed
        weight = self.quantized.float() * self.scales.float().unsqueeze(1)
        weight = weight.flatten()[:self.in_features * self.out_features]
        weight = weight.reshape(self.out_features, self.in_features).half()
        return F.linear(x, weight, self.bias)
```

At any point during inference, only the weights for the currently-executing layer are materialized in fp32. Everything else stays compressed as int8.

---

## Inference

Uses DPMSolver++ (order 2) with 20 steps — solves the diffusion ODE efficiently, achieving quality comparable to 1000-step DDPM in 20 steps.

Classifier-free guidance (CFG):
```
noise_pred = noise_uncond + scale × (noise_text - noise_uncond)
```
`guidance_scale=5.0` by default. Higher values push harder toward the prompt.

Tiled VAE decoding splits the latent into overlapping tiles and blends them — allows full-resolution decode without a large VRAM spike.

**Memory profile (384×384):**

| Stage | GPU VRAM |
|---|---|
| UNet only (quantized) | ~0.9 GB |
| During denoising loop | ~1.2–1.5 GB peak |
| VAE decode (tiled) | +~0.3 GB momentary |
| Total peak | ~1.8 GB |

---

## Stack

- `diffusers` — SD pipeline, VAE, UNet, schedulers
- `transformers` — CLIP tokenizer and text encoder
- `peft` — LoRA wrapping and merge
- `safetensors` — weight serialization
- `bitsandbytes` — (available, not used for core quant — custom implementation used instead)
- `torch` — everything else

---

## Limitations

- Fine-tuned on a small meme/character dataset — generations reflect that distribution
- Q4_K introduces quantization error; very fine details and text rendering degrade compared to fp16
- 512×512 is the upper limit on tight VRAM budgets; 384×384 is the sweet spot
- Not production-hardened — this is a research/learning project

---

## Try it

Enter a prompt. Optionally add a negative prompt. Adjust guidance scale and steps if needed. The model generates cursed images.
