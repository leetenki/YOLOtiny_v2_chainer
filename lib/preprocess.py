import argparse
import urllib.request

def download_image(): 
    parser = argparse.ArgumentParser(description='Web上から画像をダウンロードし、処理を行う。')
    parser.add_argument('--url', '-u', default='https://images-na.ssl-images-amazon.com/images/G/01/img15/pet-products/small-tiles/23695_pets_vertical_store_dogs_small_tile_8._CB312176604_.jpg', help='ダウンロードするイメージのURLを指定する')
    args = parser.parse_args()

    print('Download Image From {0} ....'.format(args.url))
    image_file_path = './sample_images/sample.jpg'
    urllib.request.urlretrieve(args.url, image_file_path)

    return image_file_path
