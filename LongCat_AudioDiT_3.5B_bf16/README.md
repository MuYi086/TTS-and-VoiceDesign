# LongCat-AudioDiT-3.5B uv service

This directory is the migrated implementation of the former LongCat
Conda API and worker. Those legacy files have been removed after the uv
migration was verified.

## Run

Install dependencies once and keep the lock file:

```bash
uv sync --project LongCat_AudioDiT_3.5B_bf16 --locked
bash start.sh
```

The service uses the final WebUI port `8323` and routes:

- `GET /v1/health`
- `POST /v1/upload_audio`
- `GET /v1/check/audio`
- `POST /v1/longCat/clone`
- `POST /internal/unload_all`

The default paths remain shared with the legacy service:

- model: `$HF_MIRROR_DIR/drbaph/LongCat-AudioDiT-3.5B-bf16`
- tokenizer: `$HF_MIRROR_DIR/google/umt5-base`
- official source: `$LONGCAT_AUDIODIT_REPO_PATH`
- prompts: `storage/clone/`; output: `storage/clone/`; designed voices remain in
  `storage/timbre/`, and a matching design upload only creates a reference map under
  `storage/timbre/.references/`; cache and lock: `storage/.cache/`

Override paths with the existing `LONGCAT_AUDIODIT_*` variables. The worker imports
the external `audiodit` package only when a synthesis request starts.

## FlashAttention decision

FlashAttention is not a LongCat dependency. The official LongCat source and the
migrated worker contain no `flash_attn` import; the model uses native
PyTorch/Transformers attention. The health endpoint reports this as
`runtime.flash_attention_policy`.

The current machine is an RTX 4070 Ti SUPER (compute capability 8.9). The local
`/home/muyi086/tts-depency/flash-attention` checkout is FlashAttention 4 beta
with FlashAttention 3/4 paths optimized for Hopper/Blackwell, so it should not be
added to this project or compiled as part of the LongCat migration. The existing
uv lock intentionally contains no FlashAttention package.
