import streamlit as st
import os
from PIL import Image
import base64
import json

from io import BytesIO

# Convert Image to Base64 
def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str

# Convert Base64 to Image
def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)

import requests
url = st.secrets["url"]
st.subheader(f'Using {len(st.session_state["exemplars_img"])} exemplar(s)')
# for f in os.listdir(str(st.session_state.id)):
#   img = Image.open(str(st.session_state.id) + '/' + f)
js = {}
query =  im_2_b64(st.session_state.query_img.convert('RGB'))
# st.text(type(str(query)))
js['query'] = "data:image/png;base64,"+str(query)[2:]
# st.text("data:image/png;base64,"+str(query)[2:])
exemplars = []
for ex in st.session_state.exemplars_img:
  ex = im_2_b64(ex)
  exemplars.append("data:image/png;base64,"+str(ex)[2:])
js['exemplars'] = exemplars
r = requests.post(url, json=js)
# x = requests.post(url, data = json)
st.text('Count: ' + str(r.json()['count']))
viz = b64_2_img(r.json()['viz'].split(',')[1])
st.image(viz)

# selected = []
# # print(type(st.session_state.query)  )
# st.image(Image.open(st.session_state.query), caption='Query Image')

# for f in os.listdir(str(st.session_state.id)):
#   if('png' in f):
#     img = Image.open(str(st.session_state.id) + '/' + f)
#     st.image(img)
#     if(st.radio(str(f), ('No', 'Use')) == 'Use'):
#       selected.append(f)

# print(selected)
    # buttons.append(st.radio(str(f), ('No', 'Use'), key=f))
# print(st.session_state.query.name)
# next = st.button('Go to next page', key='next')
# selected_exemplars = []

# for button in buttons:
#   if(button == 'Use'):
#     selected_exemplars.append(button.key)

# print(selected_exemplars)
# for i, button in enumerate(buttons):
#   if button:
#     st.write(f"{i} button was clicked")