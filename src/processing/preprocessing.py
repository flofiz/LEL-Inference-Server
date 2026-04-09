import cv2
import numpy as np
from math import sqrt
from PIL import Image
from io import BytesIO
import base64
from qwen_vl_utils import process_vision_info

def crop_and_get_boxes(image, offset = 0):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except:
        gray = image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    # Extract the largest contour (page)
    if len(contours) == 0:
        return image, (0,0)
    min_x = 10000000
    max_x = 0
    min_y = 10000000
    max_y = 0
    max_area = cv2.contourArea(contours[0])
    main_contour = [cnt  for cnt in contours if cv2.contourArea(cnt) > max_area*0.80]
    for cnt in main_contour:
        page_contour = cnt
        epsilon = 0.05 * cv2.arcLength(page_contour, True)
        page_contour = cv2.approxPolyDP(page_contour, epsilon, True)
        # Calculate the minimum and maximum x and y coordinates
        min_x = min(min(page_contour[:, :, 0].flatten())+15, min_x)
        max_x = max(max(page_contour[:, :, 0].flatten())-15, max_x)
        min_y = min(min(page_contour[:, :, 1].flatten())+15, min_y)
        max_y = max(max(page_contour[:, :, 1].flatten())-15, max_y)
    top_left = (min_x-offset, min_y-offset)
    top_right = (max_x+offset, min_y-offset)
    bottom_left = (min_x-offset, max_y+offset)
    bottom_right = (max_x+offset, max_y+offset)

    # Print the four points of the bounding box
    # Create a mask for the page contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [page_contour], 0, 255, -1)
    # Apply the mask to the original image
    page = cv2.bitwise_and(image, image, mask=mask)
    image = image[min_y:max_y, min_x:max_x]
    return image, (min_y, min_x)


def reshape_image(image):
    img_width, img_height = image.size
    max_pixels = 1280*28*28
    ratio = min(max_pixels / (img_width* img_height), 1)
    new_size = tuple(int(dim * sqrt(ratio)) for dim in image.size)
    # set each dimension to be a multiple of 28
    new_size = tuple(int(dim // 28) * 28 for dim in new_size)
    image = image.resize(new_size, Image.LANCZOS)
    new_img_width, new_img_height = image.size
    ratios = (new_img_width/img_width, new_img_height/img_height)
    return image, ratios

def cut_image(image):
    if image.size[0]>image.size[1]:
        milieu = image.size[0]/2
        image = np.array(image)
        g_image = Image.fromarray(image[:,:int(milieu*1.03),:])
        d_image = Image.fromarray(image[:,int(milieu*0.97):,:])
        images = [g_image,d_image]
        offsets = [0,int(milieu*0.97)]
    else:
        images = [image]
        offsets = [0]
    return images, offsets

def prepare_image(image):

    image = image.convert('RGB')
    images,offsets = cut_image(image)
    for image, offset in zip(images,offsets):
        # image = Image.fromarray(image)
        image, ratios = reshape_image(image)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        yield img_str, ratios, offset


def get_inputs(image, tokenizer, PROMPT, SYSTEM):
    # image = Image.open(BytesIO(base64.b64decode(image)))
    image = Image.open(BytesIO(image))
    for image_str, ratios, offset in prepare_image(image):
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": SYSTEM},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_str}"
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        # Utiliser le tokenizer pour appliquer le chat template
        # Extraire les données multimodales avec l'utilitaire Qwen
        image_inputs, video_inputs = process_vision_info(messages)
        # Appliquer le chat template
        text_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # Construire l'input pour generate()
        inputs = {
            "prompt": text_prompt,
            "multi_modal_data": {
                "image": image_inputs  # Liste d'images PIL ou tensors
            }
        }
        yield inputs, ratios, offset