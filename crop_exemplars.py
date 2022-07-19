import os
import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image

import uuid

st.set_page_config(
        page_title="Crop Exemplars",
)

st.set_option('deprecation.showfileUploaderEncoding', False)

print(st.secrets)

if 'id' not in st.session_state:
  st.session_state['id'] = uuid.uuid1()
  os.mkdir(str(st.session_state.id))



# Upload an image and set some options for demo purposes
st.header("Class Agnostic Counting Demo")
# if('query' not in st.session_state):
if 'query_img' not in st.session_state:
  img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'])
  st.session_state['query'] = img_file
else:
  img_file = st.session_state.query

# if 'query' not in st.session_state:
#   st.session_state.query = img_file

# realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
# box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
exemplars = st.sidebar.slider(label='Amount of Exemplars', min_value=1, max_value=10, value=3)

# if(img_file != None):
# if(st.session_state.query != None):
  
# if(st.session_state.query != None):
#   print(st.session_state.query.name)
# print(img_file)
st.session_state['exemplars_img'] = []
for i in range(exemplars):
  if img_file:
      img = Image.open(img_file)
      st.session_state['query_img'] = img
      # st.session_state.query = 
      # if not realtime_update:
      #     st.write("Double click to save crop")
      # Get a cropped image from the frontend
      st.subheader(f'Exemplar {i+1}')
      cropped_img = st_cropper(img, realtime_update=True, key=f'ex{i}',)
      # Manipulate cropped image at will
      st.write(f"Preview for Exemplar {i+1}")
      _ = cropped_img.thumbnail((150,150))
      st.image(cropped_img)
      st.session_state['exemplars_img'].append(cropped_img)
      # cropped_img.save(f'{st.session_state.id}/{i}.png')
  else:
    st.subheader('Image not picked!')
    break

# next = st.button('Go to next page', key='next')