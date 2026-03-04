import gradio as gr
from module.analysis import analyze_models
from ui.utils import get_model_list, get_model_path


def render_analysis_tab():
    gr.Markdown("### Model Difference Analysis")
    gr.Markdown("Compare two Safetensors models block by block.")

    with gr.Row():
        with gr.Column(scale=1):
            model_list = get_model_list()
            model_a = gr.Dropdown(label="Model A", choices=model_list)
            model_b = gr.Dropdown(label="Model B", choices=model_list)

            metric = gr.Dropdown(
                label="Comparison Metric",
                choices=[
                    "Cosine Similarity",
                    "Euclidean Distance",
                    "Mean Absolute Difference",
                ],
                value="Cosine Similarity",
            )

            analyze_btn = gr.Button("Analyze Differences", variant="primary")
            output_log = gr.Textbox(label="Log", interactive=False)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Bar Chart"):
                    output_plot_bar = gr.Plot(label="Difference Bar Chart")
                with gr.TabItem("Heatmap"):
                    output_plot_heat = gr.Plot(label="Difference Heatmap")
                with gr.TabItem("Radar Chart"):
                    output_plot_radar = gr.Plot(label="Structural Imbalance Radar")

    def run_analysis(ma, mb, met):
        if not ma or not mb:
            return None, "Please upload both Model A and Model B."

        try:
            fig_bar, fig_heat, fig_radar = analyze_models(get_model_path(ma), get_model_path(mb), metric=met)
            return fig_bar, fig_heat, fig_radar, "Analysis complete."
        except Exception as e:
            import traceback

            traceback.print_exc()
            return None, None, None, f"Error during analysis: {e}"

    analyze_btn.click(
        run_analysis,
        inputs=[model_a, model_b, metric],
        outputs=[output_plot_bar, output_plot_heat, output_plot_radar, output_log],
    )
