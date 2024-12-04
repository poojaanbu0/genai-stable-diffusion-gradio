## Prototype Development for Image Generation Using the Stable Diffusion Model and Gradio Framework

### AIM:
To design and deploy a prototype application for image generation utilizing the Stable Diffusion model, integrated with the Gradio UI framework for interactive user engagement and evaluation.

### PROBLEM STATEMENT:
Develop an accessible and user-friendly platform for generating high-quality images from text prompts using the Stable Diffusion model. The application must allow real-time user interaction, customizable settings for image generation, and facilitate evaluation and feedback from users.

### DESIGN STEPS:

#### Step 1: Set Up Environment
- Install `torch`, `diffusers`, and `gradio`.
- Use GPU (if available) for faster inference.

#### Step 2: Load Model and Define Core Function
- Load the pre-trained Stable Diffusion model (`runwayml/stable-diffusion-v1-5`) with FP16 precision.
- Define a function to generate images based on `prompt`, `num_inference_steps`, and `guidance_scale`.

#### Step 3:Build Gradio Interface
- Create a user-friendly UI with inputs for `prompt`, `num_inference_steps`, and `guidance_scale`, and an output for displaying generated images.
- Deploy the application on a local server or a cloud platform.

### PROGRAM:
```python
import torch
from diffusers import StableDiffusionPipeline
import gradio as gr



# Load the Stable Diffusion model
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe

# Initialize the pipeline
pipe = load_model()

# Define the image generation function
def generate_image(prompt, num_inference_steps=50, guidance_scale=7.5):
    """
    Generates an image based on the text prompt using Stable Diffusion.
    """
    image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image

# Set up Gradio Interface
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Stable Diffusion Image Generator")

        # Input elements
        prompt = gr.Textbox(label="Enter your prompt", placeholder="Describe the image you'd like to generate")
        num_steps = gr.Slider(10, 100, value=50, step=1, label="Number of Inference Steps")
        guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance Scale")

        # Output element
        output_image = gr.Image(label="Generated Image")

        # Button to generate image
        generate_btn = gr.Button("Generate Image")

        # Define button behavior
        generate_btn.click(fn=generate_image, inputs=[prompt, num_steps, guidance], outputs=output_image)



    demo.launch()

# Run the Gradio app
if __name__ == "__main__":
    main()

```

### OUTPUT:

![image](https://github.com/user-attachments/assets/c2afd176-4780-458e-b2da-61775c0798f2)

### RESULT:

The prototype successfully generates images based on user-provided prompts and allows interactive parameter adjustments through a user-friendly interface built with Gradio.
