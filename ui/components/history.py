import gradio as gr
from module.history import load_history
import pandas as pd


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
        load_btn = gr.Button(
            "Load Selected (Currently UI-only)", variant="primary", interactive=False
        )

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
        interactive=False,
        wrap=True,
    )

    def on_refresh():
        return get_history_df()

    refresh_btn.click(on_refresh, inputs=[], outputs=[history_table])

    # We return the load button and table so app.py can hook them up
    # if it wants to implement parameter recovery.
    return refresh_btn, load_btn, history_table
