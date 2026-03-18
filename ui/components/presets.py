import gradio as gr
from module.presets import list_presets, load_preset, save_preset
from module.error_messages import build_user_error_message


def render_presets_tab():
    gr.Markdown("### Personal Presets")
    gr.Markdown("Save and load your favorite merge configurations.")

    with gr.Row():
        with gr.Column(scale=1):
            preset_list = gr.Dropdown(label="Saved Presets", choices=list_presets())
            refresh_btn = gr.Button("Refresh List")
            load_btn = gr.Button("Load Preset", variant="primary")

        with gr.Column(scale=1):
            new_preset_name = gr.Textbox(label="Save As Name", placeholder="my_best_merge")

            # Since Gradio doesn't easily let us scrape state from another tab directly
            # without complex state management, we provide a JSON box for the user
            # to paste their current config, or we hook it up via state later.
            config_json = gr.JSON(label="Preset Configuration", value={"strategy": "mix", "velocity": 0.5})

            save_btn = gr.Button("Save Configuration", variant="primary")
            output_msg = gr.Textbox(label="Status", interactive=False)

    def on_refresh():
        return gr.update(choices=list_presets(category="default"))

    def on_load(name):
        if not name:
            return {}
        try:
            return load_preset(name, category="default")
        except Exception as e:
            return {"error": build_user_error_message(e, action="プリセット読込")}

    def on_save(name, data):
        if not name:
            return "Please enter a name."
        try:
            msg = save_preset(name, data, category="default")
            return msg
        except Exception as e:
            return build_user_error_message(e, action="プリセット保存")

    refresh_btn.click(on_refresh, inputs=[], outputs=[preset_list])
    load_btn.click(on_load, inputs=[preset_list], outputs=[config_json])
    save_btn.click(on_save, inputs=[new_preset_name, config_json], outputs=[output_msg])

    return preset_list, config_json
