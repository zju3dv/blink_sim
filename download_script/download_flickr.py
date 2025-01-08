import flickr_api as f
import sys
import os
import tqdm

FLICKR_API_KEY = ''
FLICKR_API_SECRET = ''
f.set_keys(api_key=FLICKR_API_KEY, api_secret=FLICKR_API_SECRET)


search_term = 'texture'
directory = ''
download_number = 10000

result = f.Photo.search(text=search_term, sort='relevance', per_page=100)
page_num = result.info.pages

with tqdm.tqdm(total=download_number) as pbar:
    for i in range(1, min(page_num, download_number//100)):
        result = f.Photo.search(text=search_term, sort='relevance', per_page=100, page=i)
        for photo in result.data:
            size_data = photo.getSizes()
            if 'Large' not in size_data.keys() and 'Medium 800' not in size_data.keys():
                pbar.update(1)
                continue
            pid = photo.id
            save_path = f'{directory}/{pid}'
            if os.path.exists(save_path + '.jpg') or os.path.exists(save_path + '.png'):
                pbar.update(1)
                continue

            if 'Large' in size_data.keys():
                photo.save(save_path, size_label='Large')
            else:
                photo.save(save_path, size_label='Medium 800')
            pbar.update(1)
