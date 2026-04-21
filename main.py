import gradio as gr
from detection import detect_defect

with gr.Blocks() as demo:
    gr.Markdown("# Glass Defect Detection System")
    gr.Markdown("## Detect a defect from the products.")
    gr.Markdown("Upload an image of a glass product to detect any defects.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload glass image here!")
            predict_btn = gr.Button("DETECT!")
        with gr.Column():
            output_image = gr.Image(type="pil", label="Detection result...")
            output_summary = gr.Textbox(label="summary", lines=4)
            status_box = gr.HTML(label="Status message")

    predict_btn.click(
        fn=detect_defect,
        inputs=input_image,
        outputs=[output_image, output_summary, status_box]
    )

demo.launch()