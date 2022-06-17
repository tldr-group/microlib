import requests # to get image from the web
import shutil # to save it locally
import os
import matplotlib.pyplot as plt
import numpy as np
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
        # r = requests.get(image_url, stream = True)
        r = requests.get(image_url, stream = True, verify=False)

        # Check if the image was retrieved successfully
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True
            
            # Open a local file with wb ( write binary ) permission.
            with open(filename,'wb') as f:
                shutil.copyfileobj(r.raw, f)

            img = plt.imread(filename)
            if len(img.shape) < 3:
                img = np.dstack([np.array(img)]*3 )
                print(img.shape)
            plt.imsave(filename[:-3] + 'png', img)
            os.remove(filename)
            print('Image sucessfully Downloaded: ',filename)
        else:
            print('Image Couldn\'t be retreived')
        