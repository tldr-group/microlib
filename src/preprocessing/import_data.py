import requests # to get image from the web
import shutil # to save it locally
import os
from PIL import Image
## Set up the image URL and filename
def import_data():
    f = 'data/micrographs_raw'
    if not os.path.exists(f):
        os.mkdir(f)
    for i in range(1100):
        tag = '000000' + str(i)
        tag = tag[-6:]
        image_url = f"https://www.doitpoms.ac.uk/miclib/micrographs/large/{tag}.jpg"
        filename = image_url.split("/")[-1]
        filename = f'data/micrographs_raw/{filename}'
        # Open the url image, set stream to True, this will return the stream content.
        r = requests.get(image_url, stream = True)

        # Check if the image was retrieved successfully
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True
            
            # Open a local file with wb ( write binary ) permission.
            with open(filename,'wb') as f:
                shutil.copyfileobj(r.raw, f)
            img = Image.open(filename)
            img.save(filename[:-3] + 'png')
            os.remove(filename)
            print('Image sucessfully Downloaded: ',filename)
        else:
            print('Image Couldn\'t be retreived')
        