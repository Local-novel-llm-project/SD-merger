# sd-merger

*English* | [日本語](README_ja.md)

Stable Diffusion model merging toolkit with a Reflex-based web UI and a reusable CLI pipeline.

## Features

- Merge Stable Diffusion checkpoints through the same `main.py` pipeline used by the CLI
- Queue-based execution with persistent history and rerun support
- Reflex UI organized into `ui/services`, `ui/state`, and `ui/pages`
- Extension system for custom merge algorithms and Arthemy tuning hooks
- Stable Diffusion 1.5 and SDXL support

## Installation

```bash
git clone https://github.com/Local-novel-llm-project/SD-merger
cd SD-merger
pip install -r requirements.txt
```

## Quick Start

Run the Reflex UI:

```bash
python main.py ui --port 3000
```

Run a merge directly from YAML:

```bash
python main.py merge -c example/example.yaml
```

## UI Structure

- `ui/services`: build UI-specific configs and bridge to queue/history/core pipeline
- `ui/state`: Reflex state and event handlers
- `ui/pages`: page composition for Merge, Queue, History, and Arthemy Tuner

The UI reuses the existing core implementation in `main.py`, `module/*`, and `extensions/*`.

## CLI Commands

```bash
python main.py merge -c example/example.yaml
python main.py tune --model models/example.safetensors
python main.py ui --host 0.0.0.0 --port 3000
```

## Notes

- Reflex uses port `3000` for the frontend by default. The backend is started on `3001`.
- The old Gradio UI has been removed. Advanced workflows should now be migrated by extending the Reflex pages and services.

## License

[Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)
