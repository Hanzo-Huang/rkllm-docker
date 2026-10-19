# RKLLM model-card template

Use this structure for new and refreshed Hugging Face RKLLM releases. It is
applied to the 15 `HanzoHuang/*-RKLLM` repositories as of 2026-10-19; RKLLM3 and
RKNN3 repositories are outside its scope. Match each card to its published
files, rather than copying another model's capability or validation claims.

```markdown
---
license: <upstream license identifier>
base_model: <upstream owner/model>
library_name: rkllm
pipeline_tag: <text-generation or image-text-to-text>
tags: [rkllm, rockchip, rk3576, rk3588]
---

# <Model name> · RKLLM

<Supported task> on Rockchip RK3576 and RK3588, converted from [<source>](<URL>).

<If language-only conversion of a multimodal model: prominent capability warning.>

- **Conversion toolkit:** RKLLM Toolkit <actual export version>.
- **Inputs / output:** <text / image / audio inputs → text>.
- **License:** <upstream license and link to its terms>.

## Choose your files

Use the files for your board's exact SoC. Explain available quantization options
briefly. Download size is not total runtime RAM.

| Board | Quantization | Size | Language model |
| --- | --- | --- | --- |
| <SoC> | <dtype> | <GiB> | [<filename>](<relative path>) |

<For multimodal releases: a separate compact table of matching vision/audio
encoders, one per SoC; explicitly state which companion files are required.>

## Run on your board

<One working command using a published rkllm-docker image, naming its SoC and
dtype. Link to runtime documentation for API usage and configuration.>

<Alternative hf download command with exact language and companion filenames.>

## Release notes

<Actual board results, tested variants, limits, context length if recorded, and
any conversion exception users need to reproduce the release. If no measured
results are published, say so. Do not invent performance or quality claims.>

<details>
<summary>SHA-256 checksums</summary>

<Full checksums and relative paths in a text code block.>

</details>

## Credits

<Link to original model and Rockchip conversion tools.>
```

Keep the first screen useful: task, capability limits, source, toolkit, and
available files. Put long hashes in the collapsed section. Preserve metadata,
upstream licensing, existing artifact paths, and model-specific restrictions.
Use immutable artifact revisions in Docker definitions, even when the card
shows the simpler download command.

Keep source inventories, conversion manifests, calibration records, intermediate
diagnostics, and test logs in the private job bundle. Do not publish
`source-checksums.json` or `conversion-manifest.json`. A compact `SHA256SUMS` for
deployable files may remain public. Check file links, metadata, and artifact
hashes after publishing a documentation-only change.

For a multimodal upgrade, retain existing LLM-only paths for compatibility and
publish the new matched set under a versioned directory. Clearly label both
sets in the file tables; link new Docker definitions to the validated set.
