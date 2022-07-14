import math
import os
#import sys
from typing import Iterable
from PIL import Image
import numpy as np
from torchvision.transforms import transforms

import matplotlib.pyplot as plt

import argparse

import torch

from models import build_model

from config import cfg 

def pad_to_constant(inputs, psize):
    h, w = inputs.size()[-2:]
    ph, pw = (psize-h%psize),(psize-w%psize)
    # print(ph,pw)

    (pl, pr) = (pw//2, pw-pw//2) if pw != psize else (0, 0)   
    (pt, pb) = (ph//2, ph-ph//2) if ph != psize else (0, 0)
    if (ph!=psize) or (pw!=psize):
        tmp_pad = [pl, pr, pt, pb]
        # print(tmp_pad)
        inputs = torch.nn.functional.pad(inputs, tmp_pad)
    
    return inputs

class MainTransform(object):
    def __init__(self):
        self.img_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __call__(self, img):
        img = self.img_trans(img)
        img = pad_to_constant(img, 32)
        return img

def get_scale_embedding(img, nw, nh, scale_number):
    print(img.size)
    x_dif = img.size[0]
    y_dif = img.size[1]
    scale = x_dif / nw * 0.5 + y_dif / nh * 0.5 
    scale = scale // (0.5 / scale_number)
    scale = scale if scale < scale_number - 1 else scale_number - 1
    return scale

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Class Agnostic Object Counting in PyTorch"
    )
    parser.add_argument(
        "--cfg",
        default="config/bmnet+_147.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    parser.add_argument(
        "--img",
        metavar="FILE",
        help="path to image file",
        type=str,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)

    image_path = os.path.join('test/' + args.img)
    f = args.img.split('.')[0]

    exemplars_files = os.listdir(f'test/{f}/')
    print(exemplars_files)
    query = Image.open(image_path).convert("RGB")

    query_transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

    w, h = query.size
    r = 1.0

    min_size = 384
    max_size = 1584

    if h > max_size or w > max_size:
        r = max_size / max(h, w)
    if r * h < min_size or w*r < min_size:
        r = min_size / min(h, w)
    nh, nw = int(r*h), int(r*w)
    query = query.resize((nw, nh), resample=Image.BICUBIC)

    query = MainTransform()(query)

    scale_embedding = []
    scale_number = 20

    exemplars = []
    for ex in exemplars_files:
      image_path = f"test/{f}/{ex}"
      ex = Image.open(image_path).convert("RGB")
      scale_embedding.append(get_scale_embedding(ex, nw, nh, scale_number))
      ex = query_transform(ex)
      exemplars.append(ex)

    exemplars = torch.stack(exemplars, dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg)
    checkpoint = torch.load(cfg.VAL.resume, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    scale_embedding = torch.tensor(scale_embedding)

    exemplars = {'patches': exemplars, 'scale_embedding': scale_embedding}

    exemplars['patches'] = exemplars['patches'].to(device).unsqueeze(0)
    exemplars['scale_embedding'] = exemplars['scale_embedding'].to(device)

    with torch.no_grad():
      outputs = model(query.unsqueeze(0), exemplars, False)

    density_map = outputs
    density_map = torch.nn.functional.interpolate(density_map, (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy()

    cmap = plt.cm.get_cmap('jet')

    origin_img = Image.open(f"test/{f}.jpg").convert("RGB")
    origin_img = np.array(origin_img)
    h, w, _ = origin_img.shape

    print(outputs)

    density_map = cmap(density_map / (density_map.max()) + 1e-14) * 255.0
    density_map = density_map[:,:,0:3] * 0.5 + origin_img * 0.5
    plt.title(str(outputs.sum().item()))
    plt.imshow(density_map.astype(np.uint8))
    plt.savefig('hehe.jpg')