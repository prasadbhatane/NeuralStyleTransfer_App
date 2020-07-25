import tensorflow as tf
import streamlit as st

####################################################################################
#@st.cache(persist=True)
def get_model(model_name):
	model_summary = None
	if model_name == 'VGG19':
		loaded_model = tf.keras.models.load_model("model/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")

	#model_summary = loaded_model.summary()

	return loaded_model, model_summary

###################################################################################
#loaded_model, model_summary = get_model("VGG19")
#print(model_summary)