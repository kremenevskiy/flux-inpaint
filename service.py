import os
import random
from pathlib import Path

import torch
import uvicorn
from controlnet_flux import FluxControlNetModel
from diffusers.utils import check_min_version
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from pipeline_flux_controlnet_inpaint import FluxControlNetInpaintingPipeline
from transformer_flux import FluxTransformer2DModel

check_min_version('0.30.2')


controlnet = FluxControlNetModel.from_pretrained(
    'alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Alpha', torch_dtype=torch.bfloat16
)
transformer = FluxTransformer2DModel.from_pretrained(
    'black-forest-labs/FLUX.1-dev', subfolder='transformer', torch_dtype=torch.bfloat16
)
pipe = FluxControlNetInpaintingPipeline.from_pretrained(
    'black-forest-labs/FLUX.1-dev',
    controlnet=controlnet,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to('cuda')
pipe.transformer.to(torch.bfloat16)
pipe.controlnet.to(torch.bfloat16)


def create_inpaint(
    prompt: str, image_path: str, mask_path: str, save_path: str, seed: int = 24
) -> None:
    if seed == 0:
        seed = random.randint(0, 1000)
        print('seed: ', seed)
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    generator = torch.Generator(device='cuda').manual_seed(seed)
    result = pipe(
        prompt=prompt,
        height=image.size[1],
        width=image.size[0],
        control_image=image,
        control_mask=mask,
        num_inference_steps=28,
        generator=generator,
        controlnet_conditioning_scale=0.9,
        guidance_scale=3.5,
        negative_prompt='',
        true_guidance_scale=1.0,  # default: 3.5 for alpha and 1.0 for beta
    ).images[0]

    result.save(save_path)


app = FastAPI()
MASKS_PATH = 'masks'
IMAGES_PATH = 'base-images'
OUTPUT_PATH = 'processed'

os.makedirs(MASKS_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)


@app.post('/process-image/')
async def process_image(
    prompt: str = Form(...),
    image: UploadFile = File(...),
    image_mask: UploadFile = File(...),
    seed: int = Form(0),
):
    # Ensure output directory exists

    img_path = Path(IMAGES_PATH) / image.filename
    mask_path = Path(MASKS_PATH) / image_mask.filename
    result_path = Path(OUTPUT_PATH) / image_mask.filename

    with img_path.open('wb') as file:
        file.write(await image.read())
    with mask_path.open('wb') as file:
        file.write(await image_mask.read())

    create_inpaint(
        prompt=prompt, image_path=img_path, mask_path=mask_path, save_path=result_path, seed=seed
    )

    # Return the processed file as a response
    return FileResponse(
        path=result_path,
        media_type='image/jpeg',
        filename=f'processed_{image_mask.filename}',
    )


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=11234)
