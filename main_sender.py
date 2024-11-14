import os

import requests

SECOND_API_URL = 'http://38.29.145.18:40331/process-image'

symbol = 'minor'
LOCAL_IMAGE_PATH = f'symbol_assets/{symbol}.png'
LOCAL_MASK_PATH = f'masks/{symbol}_mask.jpg'  # Path to the image you want to send
SAVE_PATH = f'{symbol}.png'  # Path to save the processed image
PROCESSED_PATH = f'{symbol}_alpha.png'


def get_prompt_from_gpt(style):
    pass


def make_white_to_alpha(img_path: str, save_path: str) -> None:
    from PIL import Image

    # Load the image
    im = Image.open(img_path).convert('RGBA')

    # Get the data of the image
    data = im.getdata()

    # Create a new data list for the image with transparency
    new_data = []
    for item in data:
        # Change all white (also shades close to white) to transparent
        # Here we define white as RGB values that are close to 255, 255, 255
        if item[0] > 200 and item[1] > 200 and item[2] > 200:  # Adjust threshold if needed
            new_data.append((255, 255, 255, 0))  # Set alpha to 0 (fully transparent)
        else:
            new_data.append(item)  # Keep original color

    # Update the image with the new data
    im.putdata(new_data)

    # Save the image as PNG with transparency
    im.save(save_path, 'PNG')


def send_prompt_with_mask(prompt: str):
    # Open the image and mask files from the specified paths
    with open(LOCAL_IMAGE_PATH, 'rb') as image_file, open(LOCAL_MASK_PATH, 'rb') as mask_file:
        # Prepare the files and data payload
        files = {
            'image': ('image.jpg', image_file, 'image/jpeg'),  # Main image file
            'image_mask': ('mask.jpg', mask_file, 'image/jpeg'),  # Mask image file
        }
        data = {'prompt': prompt}

        # Send request to the second server
        response = requests.post(SECOND_API_URL, files=files, data=data)

        if response.status_code == 200:
            # Save the processed image from the response content
            with open(SAVE_PATH, 'wb') as output_file:
                output_file.write(response.content)

            make_white_to_alpha(SAVE_PATH, PROCESSED_PATH)

            return {'message': 'Image processed and saved successfully', 'save_path': SAVE_PATH}
        else:
            return {
                'error': 'Failed to process image with mask',
                'status_code': response.status_code,
            }


# Example usage
result = send_prompt_with_mask(
    "Design the word 'MINOR' in an elegant, opulent style suited for a luxury party theme. The letters should be crafted in a sophisticated, cursive gold font with shimmering textures, reflecting light as if adorned with diamonds. Surround the text with delicate, sparkling confetti and soft, ethereal feathers, evoking a sense of glamour and festivities. The background should remain fully white, allowing the luxurious gold text and decorative elements to stand out vibrantly."
)
print(result)
