import gradio as gr
import openai
import requests
from PIL import Image
from io import BytesIO

# Add your OpenAI API key
openai.api_key = "YOUR_API_KEY"


# Text preprocessing
def preprocess_text(prompt):
    prompt = prompt.lower().strip()
    return prompt


# Image generation function
def generate_image(prompt):

    prompt = preprocess_text(prompt)

    response = openai.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024"
    )

    image_url = response.data[0].url
    img_data = requests.get(image_url).content
    image = Image.open(BytesIO(img_data))

    return image


# Gradio Interface
interface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter Image Description"),
    outputs=gr.Image(label="Generated Image"),
    title="DALL·E Text-to-Image Generator",
    description="Enter a text prompt and generate an AI image."
)

interface.launch()
