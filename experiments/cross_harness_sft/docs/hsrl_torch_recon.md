# HSRL single-4090 Torch implementation reconnaissance

Date: 2026-11-16. Static source/config inspection only. No Torch import, model construction, weight loading, CUDA call, test, synthetic run, service request, or training was performed. Code below describes a future implementation; it has not been executed. No runtime files were changed.

## Inspected installation and checkpoint

- Environment: `/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/.venv-rl-pilot`.
- Installed metadata: `torch-2.11.0+cu129`, `transformers-5.9.0`, `peft-0.21.1`, `fla_core-0.5.2`. `fla` is present; `causal_conv1d` was absent from the inspected package directory. Presence does not establish successful import, kernel availability, or device compatibility.
- Actual checkpoint directory: `/data/home/liumingxiao/cross_harness_alignment/experiments/cross_harness_sft/checkpoints/qwen35-4b-instruct-official-851bf6e`. This directory does **not** exist beneath the research worktree; use the absolute existing path.
- `config.json`: `Qwen3_5ForConditionalGeneration`, hidden size 2560, intermediate size 9216, vocabulary 248320, 32 decoder layers: 24 linear-attention layers and full attention at zero-based indices 3, 7, 11, 15, 19, 23, 27, 31. Full attention has 16 query heads, 4 KV heads, head dimension 256. Linear attention has 16 key heads and 32 value heads, dimensions 128, convolution width 4. Embedding/head weights are tied. Attention dropout is zero. EOS is 248044. Vision tower exists even for language-only work.
- Checkpoint index metadata `total_size=9319737856` bytes is the serialized tensor total, not measured runtime allocation. There is no local `generation_config.json`; override generation parameters explicitly rather than relying on inherited defaults.

## Loading and language-only LoRA

Only after the parent process has independently established that physical GPU0 is safely free, launch with `CUDA_VISIBLE_DEVICES=0`; inside this isolated process that GPU is `cuda:0`. Do not create a second rollout engine or reference model. A source-compatible loading choice is:

```python
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model

base = Qwen3_5ForConditionalGeneration.from_pretrained(
    CHECKPOINT_ABSOLUTE_PATH,
    dtype=torch.bfloat16,
    device_map={"": 0},
    attn_implementation="sdpa",
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    CHECKPOINT_ABSOLUTE_PATH, local_files_only=True,
)
base.config.use_cache = False
base.config.text_config.use_cache = False
base.requires_grad_(False)
```

Use explicit full module names. Generic suffixes or `all-linear` risk adapting the vision tower. Installed text modules are:

```python
targets = []
for i, kind in enumerate(base.config.text_config.layer_types):
    root = f"model.language_model.layers.{i}"
    targets += [f"{root}.mlp.{x}" for x in
                ("gate_proj", "up_proj", "down_proj")]
    if kind == "linear_attention":
        targets += [f"{root}.linear_attn.{x}" for x in
                    ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")]
    elif kind == "full_attention":
        targets += [f"{root}.self_attn.{x}" for x in
                    ("q_proj", "k_proj", "v_proj", "o_proj")]
    else:
        raise ValueError(kind)

policy = get_peft_model(base, LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8, lora_alpha=16, lora_dropout=0.0,
    bias="none", target_modules=targets,
    modules_to_save=None,
), autocast_adapter_dtype=True)
policy.get_base_model().gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

Rank 8 / alpha 16 is a conservative feasibility configuration, not an experimentally selected hyperparameter. This list contains 248 language projections and, from their declared dimensions, 16,232,448 LoRA parameters at rank 8. PEFT defaults promote supported low-precision adapter parameters to FP32 with `autocast_adapter_dtype=True`; frozen backbone remains BF16. Use BF16 autocast for forward/backward, FP32 adapter/master optimizer state, and an optimizer containing only `p.requires_grad` parameters. No BF16 GradScaler is needed. Do not call `policy.to(bfloat16)` afterward and unintentionally cast the trainable FP32 adapters.

Leave embeddings, tied output head, norms, linear `conv1d`, `A_log`, `dt_bias`, and every vision parameter frozen. Do not use PiSSA/OLoRA/LoftQ initialization, `bias="all"`, or `modules_to_save` here: the reference must remain the original frozen base. PEFT's ordinary default initialization leaves the effective initial policy equal to that base. Never merge adapters during the two-update run. Checkpoint construction itself can touch CUDA through FLA's fused norm initialization; even constructing this model is outside the permitted reconnaissance scope.

## Exact model access and full-prefix hidden states

For this specific standard LoRA wrapper:

```python
outer = policy.get_base_model()             # Qwen3_5ForConditionalGeneration
text = outer.model.language_model          # Qwen3_5TextModel, with injected LoRA
head = outer.lm_head                        # frozen, tied output projection
```

`get_base_model()` removes the PEFT wrapper, **not** its injected LoRA layers. `policy.get_base_model().model.language_model` is therefore correct. `outer.model` alone is the multimodal backbone; `outer.model.model.language_model` is wrong. Direct text calls bypass the wrapper's hooks, which is suitable for ordinary single-adapter LoRA, but is not a general solution for prompt tuning, mixed adapters, aLoRA, or special forward-hook features. `policy.disable_adapter()` changes the injected layers themselves and therefore still governs direct `text(...)` calls.

For one real text segment with retained prompt IDs `P` and actual generated IDs `Y`, set `X=P+Y`. Feed all of `X` once, without packing unrelated segments together. For unpadded batch size one:

```python
positions = torch.arange(X.shape[1], device=X.device).unsqueeze(0)
h = text(
    input_ids=X,
    attention_mask=torch.ones_like(X),
    position_ids=positions,
    use_cache=False,
    past_key_values=None,
    return_dict=True,
).last_hidden_state
# For a generated token at X index j, use hidden position j-1.
idx = torch.arange(len(P)-1, len(P)+len(Y)-1, device=X.device)
response_h = h[0].index_select(0, idx)
targets_y = X[0, len(P):len(P)+len(Y)]
```

An explicit 2D position tensor is accepted and expanded internally to `[4,B,T]`, containing text plus three equal rotary coordinates. For plain unpadded text this matches outer generation's position construction. It avoids mutable multimodal `rope_deltas` from previous generation. For padded batches use cumulative attention positions; a first implementation should process one real segment at a time. Do not bypass the multimodal backbone if actual image/video inputs are present. Do not concatenate tool results into generated responses: each subsequent segment gets its own actual prompt, with prior environment output as context.

## Real generation and provenance

Use `policy.generate(...)`, `policy.eval()`, `torch.no_grad()`, BF16 autocast, and a fresh cache per real segment. Supply the exact IDs from the real harness prompt encoder after its legitimate context compression, rather than decoding/re-tokenizing them or reconstructing a prompt from a completed rollout. The repository's existing `local_qwen_openai_server.py:encode_segment_prompt` uses a specific role-segment format and `add_special_tokens=False`; preserve whichever exact real prompt encoder the active harness contract specifies.

```python
generated = policy.generate(
    input_ids=prompt_ids, attention_mask=prompt_attention_mask,
    do_sample=True, temperature=TEMPERATURE,
    top_p=1.0, top_k=0, repetition_penalty=1.0,
    max_new_tokens=REMAINING_REAL_SEGMENT_BUDGET,
    eos_token_id=REAL_PROTOCOL_EOS_IDS,
    pad_token_id=TOKENIZER_PAD_OR_EOS,
    use_cache=True, logits_to_keep=1,
    return_dict_in_generate=False,
    output_scores=False, output_logits=False,
)
response_ids = generated[:, prompt_ids.shape[1]:]
```

`logits_to_keep=1` prevents full prefill vocabulary logits. Disabling stored generation scores/logits prevents accumulating one vocabulary vector per response token. Define stop conditions from the actual harness protocol; retain sampled stopping/EOS tokens as IDs, even if text extraction removes their markers. Record exact prompt IDs/hash, response IDs, temperature, stop reason, model/adapter version, task/group/segment identity and origin mask. No generation cache should survive into scoring or the optimizer step.

For exact sampling-policy ratios use the same temperature in old/current/reference selected logprobs. Full support sampling (`top_p=1`, `top_k=0`, no repetition penalties or other logits processors) makes the selected softmax logprob straightforward. If the experiment requires truncation or penalties, teacher-forced full-support logprobs are not exactly the transformed generation probabilities: either reproduce those transformations and label the distribution correctly, or explicitly report the conventional untruncated-policy objective. Do not claim equality without this distinction. Hard stop criteria decide when generation ends; they do not justify dropping already sampled tokens from provenance.

## Chunked old/reference probabilities and scalar training loss

Never call the outer full forward with default `logits_to_keep=0`, even under `no_grad()`, for a long segment. At T=16384, a `[1,T,248320]` tensor alone is about 7.58 GiB BF16 or 15.16 GiB FP32. Full `log_softmax` introduces additional storage. `logits_to_keep=1` is appropriate for generation but cannot score all response tokens.

The hidden/head split above permits projecting only generated-token predecessor states. Start with head chunks of 64 positions. In each chunk, under BF16 autocast:

```python
def selected_lp(hidden_chunk, token_chunk):
    z = head(hidden_chunk).float() / TEMPERATURE
    chosen = z.gather(-1, token_chunk[:, None]).squeeze(-1)
    return chosen - torch.logsumexp(z, dim=-1)
```

Keep the output of each no-grad chunk only as a length-C FP32 vector; release the chunk's logits before the next chunk. A 64-token vocabulary projection is about 30.3 MiB BF16 and 60.6 MiB FP32, excluding workspace and simultaneous temporaries. This bounds the output-head logits footprint; it does not guarantee the entire 16k backward fits 24 GiB.

Before any update of a rollout batch, score **old policy** with adapters enabled, eval mode, `torch.no_grad()`, full exact prefix and cache disabled. Persist detached FP32 selected logprobs per real generated token. This is a fixed denominator associated with the policy version that produced that batch. Score the **reference** similarly under:

```python
policy.eval()
with torch.no_grad(), policy.disable_adapter():
    # hidden extraction + chunked selected_lp, same exact X and temperature
    ...
```

Only the frozen base is the reference. It remains unchanged across both real updates. Keep this context completely outside any training forward/backward and exit it before restoring `policy.train()`; mutable adapter enablement must not change between checkpoint forward and recomputation. Reference score vectors may be stored on CPU and moved per segment/chunk. Old/ref hidden tensors and caches must be released before building the training graph.

For a differentiable update, call `policy.train()` with dropout zero, obtain `h` from the full-prefix, cache-free text forward, and checkpoint the **head plus scalar loss**, not the head alone. Otherwise autograd retains all chunk logits/log-softmax activations and defeats chunking. A non-reentrant checkpoint can return one scalar per head chunk:

```python
def chunk_objective(hc, yc, oldc, refc, advc, weightc):
    lp = selected_lp(hc, yc)
    ratio = torch.exp(lp - oldc)
    pg = -torch.minimum(
        ratio * advc,
        ratio.clamp(1-EPS_LOW, 1+EPS_HIGH) * advc,
    )
    delta = refc - lp
    kl = torch.exp(delta) - delta - 1.0
    return ((pg + KL_BETA * kl) * weightc).sum()

loss = sum(torch.utils.checkpoint.checkpoint(
    chunk_objective, hc, yc, oldc, refc, advc, weightc,
    use_reentrant=False,
) for hc, yc, oldc, refc, advc, weightc in chunk_inputs)
loss.backward()
```

All closure values (head, temperature, clipping, KL coefficient) must remain constant until backward completes. Pass sliced tensors as explicit checkpoint arguments; do not capture loop variables in a late-bound closure. Do not detach hidden states. Do not keep full logits for diagnostics or compute full-vocabulary entropy by default. Detached diagnostics may be computed within the head chunk. Prefer the above gather/logsumexp over creating an additional full log-softmax output.

`weightc` must implement the **already specified** objective reduction over the entire admitted batch, including all segments and gradient-accumulation microbatches. For an explicitly token-mean objective, it is origin/response mask divided by the total admitted generated-token count; sequence/group means need their own predefined weights. Do not average independently per head chunk or segment, or divide each microbatch by a different local count. Compute group-relative advantages from completed, verified real rollout groups and detach them; never fabricate differences to obtain gradients. Carry each rollout's advantage onto its generated segments. The token KL shown is the common nonnegative sampled estimator; retain the actual experiment's KL convention/coefficient if it differs. PPO clipping and KL operations belong inside the head checkpoint so their saved graph cannot retain vocabulary-sized tensors.

After all admitted segments' weighted gradients are accumulated, clip trainable adapter gradients as configured and call one optimizer step. No optimizer or adapter mode changes may happen while a live checkpoint graph still needs backward. Save adapter, optimizer, RNG, global update index and policy-version provenance. For two successive online updates, generate the second batch using the updated same model, complete genuine environment scoring/admission, and recompute its old-policy denominators before update 2. If K=2 instead denotes two optimization passes over the same batch, keep that batch's old denominators fixed; these are distinct experimental meanings and must be labeled accurately.

## Recurrent and checkpoint constraints

1. `Qwen3_5DecoderLayer` inherits `GradientCheckpointingLayer`; non-reentrant checkpointing is supported. Installed Transformers defaults to non-reentrant, but set it explicitly. `gradient_checkpointing_enable` also enables embedding-output gradients without unfreezing the embedding weight. This is appropriate for PEFT.
2. `GradientCheckpointingLayer.__call__` changes `use_cache=True` to false and discards passed `past_key_values` during training checkpointing. Training on cached prefix states would therefore be incorrect/incomplete. Always pass the entire actual segment prefix and `past_key_values=None`, `use_cache=False`.
3. `Qwen3_5GatedDeltaNet.forward` uses `chunk_gated_delta_rule` for full-prefix cache-free work. Installed FLA `chunk.py` provides a backward implementation. It uses `fused_recurrent_gated_delta_rule` only when a cache already has state and the current input has one token. Installed FLA `fused_recurrent.py:299` raises `NotImplementedError` in backward. That recurrent path is suitable for no-grad generation, not a shortcut for differentiable token-by-token training. Checkpointing cannot supply a missing recurrent backward.
4. Cache updates are stateful (`update_conv_state`, `update_recurrent_state`; single-token conv state may update in place). Replaying such mutations under gradient checkpointing is unsafe. The cache-free path avoids these mutable states.
5. If FLA is unavailable at import/runtime, Transformers' Torch chunk fallback explicitly casts q/k/v/g/beta to FP32 and contains chunk recurrences and 64-step triangular updates. Decoder checkpointing still only bounds retained inter-layer activations; one recomputed layer can have substantial temporary memory and be slow. Installed package presence does not prove that the fast path works. There is no evidence yet that long-prefix training fits this device.
6. The output-head fix eliminates the obvious full-vocabulary OOM but not full-attention workspaces, FLA activation/temporary storage, LoRA optimizer state, or vision-weight residency. SDPA avoids the known eager full attention matrix route; actual dispatch and peak memory remain unverified. Keeping a single BF16 base plus adapters is materially smaller than actor plus a separate full reference.

## Source anchors and remaining empirical gates

All paths below are relative to the environment's `lib/python3.11/site-packages/`:

- `transformers/models/qwen3_5/modeling_qwen3_5.py`: Torch chunk fallback 252–324; GatedDeltaNet 371–558; projection definitions 432–435, 646–732; decoder 757–810; text model 1153–1248; multimodal model language child 1252–1263; outer conditional generation/head 1743–1752; outer forward/head projection 1787–1880; generation preparation 1883–1956.
- `transformers/modeling_layers.py`: `GradientCheckpointingLayer` 35–93.
- `transformers/modeling_utils.py`: `gradient_checkpointing_enable` 3205 onward.
- `peft/peft_model.py`: `disable_adapter` 1023–1077; `get_base_model` 1079–1083; adapter generation 2229 onward; preparation for embedding-output grads 722–743.
- `peft/tuners/tuners_utils.py`: module target matching 2340 onward; explicit module names are matched before suffix matching.
- `peft/mapping_func.py`: `get_peft_model` 105 onward and adapter dtype option.
- `fla/ops/gated_delta_rule/fused_recurrent.py`: missing backward 299–304.
- `fla/ops/gated_delta_rule/chunk.py`: differentiable chunk function/backward around 340–417.

No runtime verification is claimed. Authorized real training, only once GPU0 is safely free, must still establish: actual kernel import/dispatch; exact real prompt/token provenance; score masks and denominators; finite old/ref/current logprobs; finite gradients and nonzero adapter change after each accepted optimizer update; preserved frozen-base parameters; successful second online update using the updated actor; persisted checkpoint/reload state and evidence; real observed allocated/reserved peaks. Preserve other workloads and existing evidence, and do not substitute a smoke/synthetic run for these real-path observations.
