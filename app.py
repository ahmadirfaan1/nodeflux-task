# from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request
import torch
import matplotlib.pyplot as plt
from models import build_model
from config import cfg
import base64
import io
import numpy as np
import matplotlib
from io import BytesIO
# import wandb

# wandb.init()

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.merge_from_file('config/test_bmnet+.yaml')
matplotlib.use('agg')

# imagenet_class_index = json.load(open('<PATH/TO/.json/FILE>/imagenet_class_index.json'))
model = build_model(cfg)
checkpoint = torch.load(cfg.VAL.resume, map_location='cpu')
model.load_state_dict(checkpoint['model'])

def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str

class MainTransform(object):
    def __init__(self):
        self.img_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def __call__(self, img):
        img = self.img_trans(img)
        img = pad_to_constant(img, 32)
        return img

def get_scale_embedding(img, nw, nh, scale_number):
    # print(img.size)
    x_dif = img.size[0]
    y_dif = img.size[1]
    scale = x_dif / nw * 0.5 + y_dif / nh * 0.5 
    scale = scale // (0.5 / scale_number)
    scale = scale if scale < scale_number - 1 else scale_number - 1
    return scale

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


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


# def get_prediction(query, exemplars):
#     with torch.no_grad():
#         outputs = model(query.unsqueeze(0), exemplars, False)

#     density_map = outputs
#     density_map = torch.nn.functional.interpolate(density_map, (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy()

#     cmap = plt.cm.get_cmap('jet')

#     origin_img = Image.open(f"test/{f}.jpg").convert("RGB")
#     origin_img = np.array(origin_img)
#     h, w, _ = origin_img.shape

#     # print(outputs)

#     density_map = cmap(density_map / (density_map.max()) + 1e-14) * 255.0
#     density_map = density_map[:,:,0:3] * 0.5 + origin_img * 0.5
#     plt.title(str(outputs.sum().item()))
#     plt.imshow(density_map.astype(np.uint8))
#     plt.savefig('hehe.jpg') 



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print(request.json)
        # file = request.files['query']
        query_b64_str = request.json['query'].split(',')[1]
        # encoded_string = base64.b64encode(bytes(request.json['query'].split(',')[1]))
        # with open("new_image.png", "wb") as new_file:
        #     new_file.write(base64.decodebytes(encoded_string))
        query_original = Image.open(io.BytesIO(base64.decodebytes(bytes(query_b64_str, "utf-8"))))
        query = query_original.copy()
        # img.save('my-image.jpeg')
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

        exemplars = []
        scale_embedding = []
        scale_number = 20
        i=0
        print(len(request.json['exemplars']))
        for exemplar in request.json['exemplars']:
            ex_b64_str = exemplar.split(',')[1]
            ex = Image.open(io.BytesIO(base64.decodebytes(bytes(ex_b64_str, "utf-8")))).convert("RGB")
            # ex.save(f'{i}.png')
            i+=1
            scale_embedding.append(get_scale_embedding(ex, nw, nh, scale_number))
            ex = query_transform(ex)
            exemplars.append(ex)

        exemplars = torch.stack(exemplars, dim=0)

        scale_embedding = torch.tensor(scale_embedding)

        exemplars = {'patches': exemplars, 'scale_embedding': scale_embedding}

        exemplars['patches'] = exemplars['patches'].to(device).unsqueeze(0)
        exemplars['scale_embedding'] = exemplars['scale_embedding'].to(device)

        with torch.no_grad():
            outputs = model(query.unsqueeze(0), exemplars, False)

        density_map = outputs
        density_map = torch.nn.functional.interpolate(density_map, (h, w), mode='bilinear').squeeze(0).squeeze(0).cpu().numpy()

        cmap = plt.cm.get_cmap('jet')

        origin_img = query_original
        origin_img = np.array(origin_img)
        h, w, _ = origin_img.shape

        # print(outputs)

        density_map = cmap(density_map / (density_map.max()) + 1e-14) * 255.0
        density_map = density_map[:,:,0:3] * 0.5 + origin_img * 0.5
        # plt.title(str(outputs.sum().item()))
        plt.imshow(density_map.astype(np.uint8))
        plt.axis(False)
        plt.savefig('hehe.jpg', transparent=True, pad_inches=0,bbox_inches='tight')

        query = im_2_b64(Image.open('hehe.jpg'))
        # st.text(type(str(query)))
        query = "data:image/png;base64,"+str(query)[2:]
        # print(request.files['query'])
        # print(request.files['exemplars'])
        # img_bytes = file.read()
        # class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'count': outputs.sum().item(), 'viz': query})
        # return jsonify({'class_id': class_id, 'class_name': class_name})

@app.route("/")
def hello_world():
    return "<p>Test</p>"

if __name__ == '__main__':
    app.run(debug=True, port=8080, threaded=True, host=('0.0.0.0'))