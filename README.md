# sd-merger

*English* | [日本語](README_ja.md)

A command-line tool for merging Stable Diffusion models with flexible strategies.


## Features

- Multiple merge strategies: subtraction, addition, multiplication, average, replace
- Block-level merging with MBW (Merge Block Weight) support
- Extension system for custom merge algorithms
- LoRA extraction and merging capabilities
- Stable Diffusion 1.5 and SDXL support


## Installation

```bash
git clone https://github.com/Local-novel-llm-project/SD-merger
cd SD-merger
pip install -r requirements.txt
```

## Quick Start

```bash
python main.py -c example/example.yaml
```

## Configuration

Create a YAML config file:

```yaml
target_model: "base_model"
models:
  - left: "model_a"
    right: "model_b"
    velocity: 1.0
    strategy: "addition"
    key_patterns:
      - "."
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `target_model` | Base model to merge onto |
| `left` / `right` | Models to merge |
| `velocity` | Merge strength (0.0-1.0) |
| `strategy` | Merge algorithm |
| `key_patterns` | Layers to target |

### Available Strategies

- `subtraction` - Calculate difference
- `addition` - Add models
- `multiplication` - Multiply weights
- `average` - Blend models
- `replace` - Direct replacement

## Extensions

Extensions add extra functionality:

| Extension | Description |
|-----------|-------------|
| `supermerger_mbw` | Merge Block Weight for layer-wise control |
| `lora_ops` | LoRA extraction and merging |
| `resize_lora` | Resize LoRA ranks |
| `quantum_merge` | Advanced merge algorithms |

See [README_ja.md](README_ja.md) for extension details.

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
