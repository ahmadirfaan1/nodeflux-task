import os
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

import uuid

st.set_page_config(
        page_title="Crop Exemplars",
)

st.set_option('deprecation.showfileUploaderEncoding', False)


if 'id' not in st.session_state:
  st.session_state['id'] = uuid.uuid1()
  os.mkdir(str(st.session_state.id))

st.header("Class Agnostic Counting Demo")
st.text("Refresh the tab if you want to change image")

if 'query_img' not in st.session_state:
  img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
  st.session_state['query'] = img_file
else:
  img_file = st.session_state.query


exemplars = st.sidebar.slider(label='Amount of Exemplars', min_value=1, max_value=10, value=3)

st.session_state['exemplars_img'] = []
for i in range(exemplars):
  if img_file:
      img = Image.open(img_file)
      st.session_state['query_img'] = img

      st.subheader(f'Exemplar {i+1}')
      cropped_img = st_cropper(img, realtime_update=True, key=f'ex{i}',)
      st.write(f"Preview for Exemplar {i+1}")
      _ = cropped_img.thumbnail((150,150))
      st.image(cropped_img)
      st.session_state['exemplars_img'].append(cropped_img)
  else:
    st.subheader('Image not picked!')
    break

