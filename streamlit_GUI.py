import streamlit as st
from nst import start_training
from PIL import Image
import numpy as np

st.markdown("# Neural Style Transfer App")
st.sidebar.markdown("# Menu")
start_flag = False

c_img = st.sidebar.file_uploader("Upload Content Image", type=['jpg', 'jpeg', 'png'])
s_img = st.sidebar.file_uploader("Upload Style Image", type=['jpg', 'jpeg', 'png'])

if c_img != None and s_img != None:
	c_img = Image.open(c_img)
	s_img = Image.open(s_img)

	c_img = c_img.resize((224,224))
	s_img = s_img.resize((224,224))

	c_img = np.array(c_img)[:,:,:3]
	s_img = np.array(s_img)[:,:,:3]

	#st.write(c_img.shape)
	#st.write(s_img.shape)
	#c_img = tf.convert_to_tensor(c_img[:,:,:3])
	#s_img = tf.convert_to_tensor(s_img[:,:,:3])

	start_flag = True

if start_flag and c_img.any() != None and s_img.any() != None and st.sidebar.checkbox("Show Content and Style Images", True, key='cs_img'):
	st.markdown("### Content Image :")
	st.image(c_img)
	st.markdown("### Style Image :")
	st.image(s_img)
    

model_name = st.sidebar.selectbox("Choose Model :", ['VGG19'], key='model_name')
ITERATIONS = int(st.sidebar.number_input("Number of iterations", value=20, min_value=1, max_value=50, key='ITERATIONS'))
LEARNING_RATE = float(st.sidebar.number_input("Learning Rate", value=7.0, min_value=0.0000001, max_value=20.0, key='LEARNING_RATE'))
W_C = float(st.sidebar.number_input("Content Weight", value=10.0, min_value=1.0, max_value=100.0, key='W_C'))
W_S = float(st.sidebar.number_input("Style Weight", value=20.0, min_value=1.0, max_value=100.0, key='W_S'))


if start_flag and c_img.any() != None and s_img.any() != None and st.sidebar.button("Generate NST Image", key='train'):
	st.markdown("Generating............")
	st.write("Wait for few minutes...")
	
	generated_img = start_training(c_img, s_img, ITERATIONS, W_C, W_S)
	st.image(generated_img)