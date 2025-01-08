import requests
import os
import shutil
import tqdm
import urllib.parse
from PIL import Image
from io import BytesIO

# define your Pixabay API key
PIXABAY_API_KEY = 'your_pixabay_api_key'

# define the search term
search_term = 'texture'

# define the directory where you want to save the images
directory = './pixabay_images/'

# create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# define the number of images you want to download
num_images = 10000

# calculate the number of pages needed (20 images per page)
num_pages = num_images // 20

with tqdm.tqdm(total=num_images) as pbar:
    # download each page of images
    for page in range(1, num_pages + 1):
        # define the url
        url = f"https://pixabay.com/api/?key={PIXABAY_API_KEY}&q={search_term}&image_type=photo&per_page=20&page={page}"

        # send a GET request
        response = requests.get(url)

        # get the JSON data from the response
        data = response.json()

        # get the image details
        images = data['hits']

        # download each image
        for i, image in enumerate(images, start=(page-1)*20):
            url = image['largeImageURL']  # use largeImageURL instead of webformatURL

            # Parse the url to get the extension
            pid = image['id']
            parsed = urllib.parse.urlparse(url)
            ext = os.path.splitext(parsed.path)[1]

            save_path = directory + f'/{pid}{ext}'
            if os.path.exists(save_path):
                pbar.update(1)
                continue

            # send a GET request to the image url
            response = requests.get(url, stream=True)

            # check that the request was successful
            if response.status_code == 200:
                # Load image with PIL and check if it's RGBA
                img = Image.open(BytesIO(response.content))
                if img.mode == 'RGBA':
                    pbar.update(1)
                    continue

                # Open a file and save the image
                with open(save_path, 'wb') as out_file:
                    out_file.write(response.content)
                    pbar.update(1)

