import gradio as gr
from ui.utils import get_model_list

def create_model_dropdown_pair():
    """Returns a pair of Gradio dropdowns for Model A and Model B."""
    model_list = get_model_list()
    model_a = gr.Dropdown(label="Model A (Left)", choices=model_list)
    model_b = gr.Dropdown(label="Model B (Right)", choices=model_list)
    return model_a, model_b
