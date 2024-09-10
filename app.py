import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np

from keras.models import load_model
model = load_model('Conv_model_1.keras', compile=False)

st.title("Welcome to Handwritten Digit Recognizer")

st.text("Draw a Digit Below")
canvas = st_canvas(
    fill_color="rgba(255,165,0,0.3)",
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key='canvas'
)


btn = st.button("Recognize")

if btn:
    img = canvas.image_data
    
    image = Image.fromarray(img.astype('uint8'),'RGBA')
    image = image.resize((28,28))
    image = image.convert('L') # convert to grayscale
    
    np_image = np.array(image)
    # np_image = np_image/255.0
    # np_image = 1 - np_image
    
    np_image = np_image.reshape(1, 28,28,1)
    pred = model.predict(np_image)
    predicted_digit = np.argmax(pred)
    print(pred)
    print(predicted_digit)
    st.subheader(f"It looks like: ")
    st.header(predicted_digit)
    
    
footer_html = """
<div style='text-align: center;'>
  <p>Developed with ❤️ using Streamlit</p>
  <p>By Sumit Kumar</p>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)

