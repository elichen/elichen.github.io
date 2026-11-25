Co-created with AI

## Local model files

The published demos load their weights from Hugging Face. For training or parity tools that read files from disk, run this from the repository root, replacing `snake` with the demo directory you need.

```sh
hf download elichen77/personal-site-models --revision e4f7f82c373c5c56b61d21944b77b567f8f1c18a --include "snake/*" --local-dir .
```
