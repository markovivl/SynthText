import requests
from urllib.parse import urlencode
import os
import json


BASE_URL = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'

def download_json():
    public_key = 'https://disk.yandex.ru/d/8oABHwurLBtrhA'  
    # get link
    final_url = BASE_URL + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    download_url = response.json()['href']

    # download json with links
    download_response = requests.get(download_url)
    if not os.path.exists('./dataset'):
        os.mkdir('./dataset')
    with open('./dataset/all_archives.json', 'wb') as f: 
        f.write(download_response.content)

        
def download_data(split='train'):
    with open('./dataset/all_archives.json', 'rb') as f:
        all_archives = json.load(f)
    split_set = all_archives[split]
    if not os.path.exists(f'./dataset/{split}'):
        os.mkdir(f'./dataset/{split}')
    for k, v in split_set.items():
        public_key = v
        final_url = BASE_URL + urlencode(dict(public_key=public_key))
        response = requests.get(final_url)
        download_url = response.json()['href']
        download_response = requests.get(download_url)
        with open(f'./dataset/{split}/archive_{k}.tar.gz', 'wb') as f:
            f.write(download_response.content)
    

if __name__ == "__main__":
    download_json()
    download_data(split='test')
    #download_data(split='train')
        