# RSIHub × HarnessRisk adapter image

This image packages the pinned RSIHub and HarnessRisk source trees, the official case-identity manifest, the receipt-to-task-vector adapter, and a Hugging Face model downloader. It deliberately excludes HarnessRisk case contents, model weights, run outputs, and API credentials.

## Scope

The image is a reproducible adapter/runtime bundle, not a claim that the full RSIHub search lifecycle is already integrated. The current project pilot validated one official HarnessRisk rule receipt through RSIHub's task-vector normalizer. It did **not** run AHE, Hyperagents, A-Evolve, GEPA, the LLM judge, or the full matrix. See `experiments/cross_harness_sft/docs/rsihub_harnessrisk_baseline_20261126.md` in the repository for the protocol and remaining lifecycle gate.

The pinned upstream sources are available at `/opt/RSIHub` and `/opt/HarnessRisk`. The HarnessRisk case data is not shipped by its source repository; mount the separately authorized dataset under `/data/HarnessRisk`. The frozen case identity manifest is at `/opt/harnessrisk_case_manifest.json`.

## Models

The experiment's registered 4B H0 is `Qwen/Qwen3.5-4B` at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Do not silently substitute the distinct pretrained-only repository `Qwen/Qwen3.5-4B-Base`.

The full merged BF16 GRPO-S step-200 checkpoint is in the private Hugging Face repository `Limax11/harness_alignment`, at immutable commit `f8ca0ae2af5f52527572e933f03f5bfa6b8df5cd`, path `online_rl/a6000_r1_lora_2g8_300_20261121/step_0200_merged`. It is a full model, so no LoRA adapter needs to be attached. Set `HF_TOKEN` in the environment for this private repository. The downloader verifies the pinned `SHA256SUMS` identity and every listed file before it reports success.

```sh
read -s GHCR_TOKEN  # GitHub PAT with read:packages, authorized for this package
printf '%s' "$GHCR_TOKEN" | docker login ghcr.io -u <github-user> --password-stdin
unset GHCR_TOKEN
mkdir -p ./models/base4b ./models/grpo200
docker run --rm -v "$PWD/models:/models" ghcr.io/limax666/rsihub-harnessrisk-adapter:latest \
  download-model base4b --out /models/base4b
read -s HF_TOKEN; export HF_TOKEN
docker run --rm -e HF_TOKEN -v "$PWD/models:/models" ghcr.io/limax666/rsihub-harnessrisk-adapter:latest \
  download-model grpo200 --out /models/grpo200
unset HF_TOKEN
```

The GRPO model will be under `./models/grpo200/online_rl/a6000_r1_lora_2g8_300_20261121/step_0200_merged`. Keep tokens out of command history and logs. The official base is public; the GRPO repository requires HF read access.

## Receipt adapter

Mount a directory containing a complete official HarnessRisk JSON receipt and an output directory. The converter refuses incomplete receipts and refuses to overwrite prior artifacts:

```sh
docker run --rm \
  -v "$PWD/receipts:/receipts:ro" \
  -v "$PWD/converted:/converted" \
  ghcr.io/limax666/rsihub-harnessrisk-adapter:latest \
  adapt-receipt /receipts/receipt.json --task-id action_041 --trial 0 --out /converted/action_041
```

It writes RSIHub `task_vector.json`, the four separate official rule metrics, and artifact hashes. Infrastructure failures are not converted to a zero reward.

## Image references

Use the immutable `sha-<Git commit>` tag for a run record. `latest` tracks the default branch. Both tags resolve to a GHCR image built by the repository's `Publish RSIHub-HarnessRisk adapter image` workflow. The image contains CPU-side adapter tools; model inference still needs a compatible host serving stack / GPU runtime and the actual HarnessRisk lifecycle runner. Do not treat this bundle by itself as a completed four-condition evaluation.

GHCR currently creates the package as private. A different server must have package read permission and log in with a GitHub token carrying `read:packages`; alternatively, build the image from this public repository branch. The HF GRPO checkpoint is separately private and requires its own HF read token.
