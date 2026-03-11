import gradio as gr
from module.history import load_history, export_recipe
from module.error_messages import build_user_error_message
import pandas as pd
import os
import tempfile


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

    data = []
    for h in history:
        cfg = h.get("config", {})
        models = cfg.get("models", [{}])[0]
        data.append(
            {
                "Date": h.get("date", ""),
                "Output Name": h.get("output_name", ""),
                "Strategy": models.get("strategy", ""),
                "Velocity": models.get("velocity", ""),
                "Model A": models.get("left", ""),
                "Model B": models.get("right", ""),
                "Status": h.get("status", "Unknown"),
            }
        )
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

    history_table.select(on_select, None, [rerun_btn, download_btn, selected_index])

    refresh_btn.click(on_refresh, inputs=[], outputs=[history_table])

    def on_download(idx):
        if idx < 0:
            return None
        history = load_history()
        if idx >= len(history):
            return None
        entry = history[idx]

        # Create temp yaml
        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        export_recipe(entry, path)
        return path

    download_btn.click(on_download, inputs=[selected_index], outputs=[download_btn])

    def on_rerun(idx):
        if idx < 0:
            return "No row selected."
        history = load_history()
        if idx >= len(history):
            return "Invalid row selected."
        entry = history[idx]

        fd, path = tempfile.mkstemp(suffix=".yaml")
        os.close(fd)
        export_recipe(entry, path)

        from module.queue_manager import queue_manager

        try:
            task_id = queue_manager.add_task(entry["config"], entry["output_name"], task_name="History Re-run")
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

        from module.queue_manager import queue_manager
        import yaml

        try:
            with open(file.name, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            out_name = config.get("output_name", "imported_recipe_merge.safetensors")
            if "models" in config and len(config["models"]) > 0 and "output_name" in config["models"][0]:
                out_name = config["models"][0]["output_name"]

            task_id = queue_manager.add_task(config, out_name, task_name="Imported Recipe")
            return f"Imported recipe task '{task_id}' added to queue. Output will be {out_name}"
        except Exception as e:
            return build_user_error_message(e, action="インポートしたレシピの登録")

    import_run_btn.click(on_run_imported, inputs=[import_file], outputs=[output_log])

    return refresh_btn, rerun_btn, history_table
