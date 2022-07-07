---
language: en
tags:
- pythae
license: apache-2.0
---

### Downloading this model from the Hub
This model was trained with pythae. It can be downloaded or reloaded using the method `load_from_hf_hub`
```python
>>> from pythae.models import AutoModel
>>> my_downloaded_vae = AutoModel.load_from_hf_hub(hf_hub_path="path/to/this/repo")
```