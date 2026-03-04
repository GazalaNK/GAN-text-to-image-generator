import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
import gradio as gr
from PIL import Image

model_id = "openai/clip-vit-base-patch32"

tokenizer = CLIPTokenizer.from_pretrained(model_id)
text_encoder = CLIPTextModel.from_pretrained(model_id)

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def get_text_embedding(prompt):
  tokens = preprocess_text(prompt)

  with torch.no_grad():
    embedding = text_encoder(**tokens)
    return embedding.last_hidden_state

def generate_image(prompt):

    # create CLIP embeddings
    embeddings = get_text_embedding(prompt)

    # generate image using Stable Diffusion
    image = pipe(prompt).images[0]

    return image


def interface(prompt):

    image = generate_image(prompt)

    return image


demo = gr.Interface(
    fn=interface,
    inputs=gr.Textbox(label="Enter Text Prompt"),
    outputs=gr.Image(label="Generated Image"),
    title="Text to Image Generator",
    description="Stable Diffusion + CLIP + Hugging Face"
)

demo.launch()
