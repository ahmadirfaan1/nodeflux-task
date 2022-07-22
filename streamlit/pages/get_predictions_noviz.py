import streamlit as st
import base64
import time

from io import BytesIO

st.set_page_config(
        page_title="Get Predictions",
)

# Convert Image to Base64 
def im_2_b64(image):
    buff = BytesIO()
    image.save(buff, format="JPEG")
    img_str = base64.b64encode(buff.getvalue())
    return img_str

import requests
url = st.secrets["url"]
if("query_img" not in st.session_state):
    st.subheader("Error, check your exemplars or the service is broken!")
else:
    st.subheader(f'Using {len(st.session_state["exemplars_img"])} exemplar(s)')

    js = {}
    query =  im_2_b64(st.session_state.query_img.convert('RGB'))

    js['query'] = "data:image/png;base64,"+str(query)[2:]

    exemplars = []
    for ex in st.session_state.exemplars_img:
        ex = im_2_b64(ex)
        exemplars.append("data:image/png;base64,"+str(ex)[2:])
    js['exemplars'] = exemplars
    start_time = time.time()
    r = requests.post(url, json=js)
    end_time = time.time()

    st.text('Count: ' + str(r.json()['count']))
    st.text('Time taken from request to result: ' + str(end_time-start_time))
