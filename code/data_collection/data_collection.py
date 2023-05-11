import os
import requests
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ['GOOGLE_API_KEY']
CSE_ID = os.environ['GOOGLE_CSE_ID']

def save_image(url, image_number, size, folder):
    try:
        response = requests.get(url)
        img_pil = Image.open(BytesIO(response.content))
        img_resized = img_pil.resize((size, size), Image.LANCZOS)

        target_file = os.path.join(folder, f'image_{image_number:04}.jpg')
        img_resized.save(target_file, 'JPEG')

    except Exception as e:
        print(f'Error downloading and resizing image: {url} {str(e)}')

def get_last_saved_image_number(folder):
    image_files = [f for f in os.listdir(folder) if f.endswith('.jpg')]
    if not image_files:
        return 0

    image_numbers = [int(f.split('_')[1].split('.')[0]) for f in image_files]
    return max(image_numbers)

def get_image_urls(api_key, cse_id, search_term, start_index):
    url = f'https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={search_term}&searchType=image&start={start_index}&num=10'
    response = requests.get(url)
    data = response.json()

    image_urls = []
    if 'items' in data:
        for item in data['items']:
            image_urls.append(item['link'])

    return image_urls

def collect_data(query, size, folder, images):
    if not os.path.exists(folder):
        os.makedirs(folder)
    last_saved_image_number = get_last_saved_image_number(folder)
    print(f'Starting from image number: {last_saved_image_number + 1}')

    image_number = last_saved_image_number
    start_index = 1

    while image_number < images:
        image_urls = get_image_urls(API_KEY, CSE_ID, query, start_index)

        if not image_urls:
            break

        for image_url in image_urls:
            image_number += 1
            save_image(image_url, image_number, size, folder)
            print(f'Saved image {image_number}: {image_url}')

            if image_number >= images:
                break

        start_index += 10
    print("finished collecting images")