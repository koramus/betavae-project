import os
import os.path
from PIL import Image, UnidentifiedImageError
import numpy as np

# Resizes, greyscales and packs chair images into a NumPy file

dataset_path = '/tmp/rendered_chairs'

data = []
for root, dirs, files in os.walk(dataset_path):
    for name in files:
        path = os.path.join(root, name)
        try:
            with Image.open(path) as im:
                im = im.convert('L')
                im = im.resize((64, 64), Image.BILINEAR)

                array = np.asarray(im)
                array = array / 255

                # add grayscale 'channel'
                array = np.expand_dims(array, 0)

                data.append(array)
        except FileNotFoundError:
            pass
        except UnidentifiedImageError:
            pass

data = np.stack(data)
np.save(os.path.join('.', 'chairs'), data)
