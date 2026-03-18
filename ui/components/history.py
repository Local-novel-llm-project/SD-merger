import gradio as gr
from module.history import load_history, export_recipe
from module.error_messages import build_user_error_message
import pandas as pd
import os
import tempfile
from ui.utils import enqueue_merge_task


def _get_primary_model_config(config: dict) -> dict:
    models = config.get("models")
    if not isinstance(models, list):
        return {}
    for model in models:
        if isinstance(model, dict):
            return model
    return {}


def _build_history_row(entry: dict) -> dict:
    config = entry.get("config", {})
    model_config = _get_primary_model_config(config)

    strategy = model_config.get("strategy", "")
    velocity = model_config.get("velocity", "")
    model_a = model_config.get("left", "")
    model_b = model_config.get("right", "")

    if not model_config and config.get("target_model"):
        strategy = "target-only"
        model_a = config.get("target_model", "")

    return {
        "Date": entry.get("date", ""),
        "Output Name": entry.get("output_name", ""),
        "Strategy": strategy,
        "Velocity": velocity,
        "Model A": model_a,
        "Model B": model_b,
        "Status": entry.get("status", "Unknown"),
    }


def _resolve_imported_output_name(
    config: dict,
    default_output_name: str = "imported_recipe_merge.safetensors",
) -> str:
    output_name = config.get("output_name")
    if output_name:
        return str(output_name)

    model_config = _get_primary_model_config(config)
    if model_config.get("output_name"):
        return str(model_config["output_name"])

    return default_output_name


def _cleanup_download_file(path: str | None) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _create_download_recipe_file(entry: dict) -> str:
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    export_recipe(entry, path)
    return path


def get_history_df():
    history = load_history()
    if not history:
        return pd.DataFrame(
            columns=[
                "Date",
                "Output Name",
                "Strategy",
                "Velocity",
                "Model A",
                "Model B",
                "Status",
            ]
        )

    data = [_build_history_row(entry) for entry in history]
    return pd.DataFrame(data)


def render_history_tab():
    gr.Markdown("### Merge History")
    gr.Markdown(
        "View your past merges. Select a row and click 'Load Selected' to restore the parameters into the main Merge Models tab."
    )

    with gr.Row():
        refresh_btn = gr.Button("Refresh History", variant="secondary")
        rerun_btn = gr.Button("Re-run Selected Merge", variant="primary", interactive=False)
        download_btn = gr.DownloadButton("Download Recipe (YAML)", interactive=False)

    with gr.Row():
        import_file = gr.File(label="Import Recipe (YAML)", file_types=[".yaml", ".yml"])
        import_run_btn = gr.Button("Run Imported Recipe", variant="primary", interactive=False)

    output_log = gr.Textbox(label="Action Log", interactive=False)

    history_table = gr.Dataframe(
        value=get_history_df(),
        headers=[
            "Date",
            "Output Name",
            "Strategy",
            "Velocity",
            "Model A",
            "Model B",
            "Status",
        ],
        interactive=False,  # to allow row selection we need gr.Dataframe selection event
        wrap=True,
    )

    def on_refresh():
        return get_history_df()

    def on_select(evt: gr.SelectData):
        # evt.index is a tuple [row, col]
        return gr.update(interactive=True), gr.update(interactive=True), evt.index[0]

    selected_index = gr.State(-1)
    download_path_state = gr.State(None)

    history_table.select(on_select, None, [rerun_btn, download_btn, selected_index])

    refresh_btn.click(on_refresh, inputs=[], outputs=[history_table])

    def on_download(idx, previous_path):
        _cleanup_download_file(previous_path)
        if idx < 0:
            return None, None
        history = load_history()
        if idx >= len(history):
            return None, None
        entry = history[idx]
        path = _create_download_recipe_file(entry)
        return path, path

    download_btn.click(
        on_download,
        inputs=[selected_index, download_path_state],
        outputs=[download_btn, download_path_state],
    )

    def on_rerun(idx):
        if idx < 0:
            return "No row selected."
        history = load_history()
        if idx >= len(history):
            return "Invalid row selected."
        entry = history[idx]

        try:
            task_id = enqueue_merge_task(
                entry["config"],
                entry["output_name"],
                task_name="History Re-run",
            )
            return f"Re-run task '{task_id}' added to queue. Output will be {entry['output_name']}"
        except Exception as e:
            return build_user_error_message(e, action="履歴レシピの再実行登録")

    rerun_btn.click(on_rerun, inputs=[selected_index], outputs=[output_log])

    def on_import_upload(file):
        if file is None:
            return gr.update(interactive=False)
        return gr.update(interactive=True)

    import_file.upload(on_import_upload, inputs=[import_file], outputs=[import_run_btn])
    import_file.clear(lambda: gr.update(interactive=False), inputs=[], outputs=[import_run_btn])

    def on_run_imported(file):
        if file is None:
            return "No file uploaded."

        import yaml

        try:
            with open(file.name, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            out_name = _resolve_imported_output_name(config)

            task_id = enqueue_merge_task(
                config,
                out_name,
                task_name="Imported Recipe",
            )
            return f"Imported recipe task '{task_id}' added to queue. Output will be {out_name}"
        except Exception as e:
            return build_user_error_message(e, action="インポートしたレシピの登録")

    import_run_btn.click(on_run_imported, inputs=[import_file], outputs=[output_log])

    return refresh_btn, rerun_btn, history_table
