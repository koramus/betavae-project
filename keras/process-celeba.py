import os
import os.path
from PIL import Image, UnidentifiedImageError
import numpy as np

# Crops and resizes CelebA images

dataset_path = '/tmp/img_align_celeba'
output_path = '/tmp/img_align_celeba2'

for root, dirs, files in os.walk(dataset_path):
    for name in files:
        path = os.path.join(root, name)
        try:
            with Image.open(path) as im:
                im = im.crop((0, 35, 178, 213)) # 178x178 aligned with the face
                im = im.resize((64, 64), Image.BILINEAR)

                im.save(os.path.join(output_path, name))
        except FileNotFoundError:
            pass
        except UnidentifiedImageError:
            pass
