# sd-merger

[English](README.md) | *日本語*

Stable Diffusionモデルのマージを行うコマンドラインツールです。


## 特徴

- 複数のマージ戦略: subtraction, addition, multiplication, average, replace
- MBW (Merge Block Weight) によるレイヤー単位のマージ
- 拡張機能システムによるカスタムマージアルゴリズム
- LoRAの抽出・LoRA同士のマージ・モデルへの適用機能
- Stable Diffusion 1.5 と SDXL に対応


## インストール

```bash
git clone https://github.com/Local-novel-llm-project/SD-merger
cd SD-merger
pip install -r requirements.txt
```

## 使い方

```bash
python main.py -c example/example.yaml
```

## 設定

YAML設定ファイルを作成:

```yaml
target_model: "ベースモデル"
models:
  - left: "モデルA"
    right: "モデルB"
    velocity: 1.0
    strategy: "addition"
    key_patterns:
      - "."
```

### 主要パラメータ

| パラメータ | 説明 |
|-----------|------|
| `target_model` | マージ先のベースモデル |
| `left` / `right` | マージするモデル |
| `velocity` | マージ強度 (0.0-1.0) |
| `strategy` | マージアルゴリズム |
| `key_patterns` | 対象レイヤー |

### 利用可能な戦略

- `subtraction` - 差分を計算
- `addition` - モデルを加算
- `multiplication` - 重みを乗算
- `average` - モデルをブレンド
- `replace` - 直接置換

## 拡張機能

拡張機能で追加機能を有効化:

| 拡張機能 | 説明 |
|---------|------|
| `supermerger_mbw` | レイヤー単位のMerge Block Weight制御 |
| `lora_ops` | LoRAの抽出・マージ・モデル適用 |
| `resize_lora` | LoRAランクのリサイズ |
| `quantum_merge` | 高度なマージアルゴリズム |

### 拡張機能の開発

`extensions/` ディレクトリに独自のフォルダを作成し、`__init__.py` に `setup()` 関数を定義:

```python
from module.extension_manager import register_strategy, register_pre_merge_hook
from sd_mecha import merge_method, Parameter, Return

@merge_method
def my_strategy(a, b, velocity=1.0, **kwargs):
    return (a + b) * velocity * 0.5

def setup():
    register_strategy("my_algorithm", my_strategy)
```

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
