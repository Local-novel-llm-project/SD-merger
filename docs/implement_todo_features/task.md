# SD-merger TODO List

- [x] UI: LoRA Operations (Extract & Merge LoRAs) の UI 実装 (`ui/app.py`)
- [ ] Optimization: モデル操作時のメモリ消費量の最適化 (`extensions/lora_ops/kohyas/model_util.py`)
- [ ] Feature: UNet/SDXL UNet Hypernetworks サポートの実装 (`original_unet.py`, `sdxl_original_unet.py`)
- [ ] Feature: LoRA マージ時の `sai_model_spec` 読み込みサポート (`svd_merge_lora.py`, `merge_lora.py`)
- [ ] Optimization: `sai_model_spec.py` のハッシュ計算におけるメモリ最適化
- [ ] Optimization: `train_util.py` 内の特定の処理の高速化
- [ ] Testing: `text_encoder` 指定形式の `dtype` で正しく作成できるかの検証 (`model_util.py`)
