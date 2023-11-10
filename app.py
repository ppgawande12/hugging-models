from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from transformers import set_seed

class CFG:
    device = "cuda"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12


set_seed(CFG.seed)

def load_image_gen_model():
    try:
        image_gen_model = StableDiffusionPipeline.from_pretrained(
            CFG.image_gen_model_id,
            torch_dtype=torch.float16,
            revision="fp16",
            use_auth_token='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',
            guidance_scale=CFG.image_gen_guidance_scale
        )
        image_gen_model.to(CFG.device)
        return image_gen_model
    except Exception as e:
        print(f"Error loading image generation model: {e}")
        return None

def generate_image(prompt, model):
    try:
        image = model(
            prompt,
            num_inference_steps=CFG.image_gen_steps,
            generator=CFG.generator,
            guidance_scale=CFG.image_gen_guidance_scale
        ).images[0]

        image = image.resize(CFG.image_gen_size)
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def main():
    image_gen_model = load_image_gen_model()

    if image_gen_model:
        prompt = "astronaut in space"
        generated_image = generate_image(prompt, image_gen_model)

        if generated_image:
            generated_image.show()

if __name__ == "__main__":
    main()
