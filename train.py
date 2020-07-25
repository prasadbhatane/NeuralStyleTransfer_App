import tensorflow as tf
#import tensorflow.GradientTape as tape
import matplotlib.pyplot as plt
from content_style import content_cost,style_cost
from processing_img import show_image
import streamlit as st


ITERATIONS = 20
LEARNING_RATE = 7.0
W_C = 10.0
W_S = 20.0
image_gallery = []

####################################################################
def get_image_gallery():
	plt.figure(figsize = (12, 12))

	for i in range(10):
	    plt.subplot(5, 2, i + 1)
	    show_image(generated_images[i])
	plt.show()
	return image_gallery

####################################################################
def train_nst(c_img, s_img, iter=ITERATIONS, wc=W_C, ws=W_S):
	gen_img = tf.Variable(c_img, dtype = tf.float32)
	optimizer = tf.optimizers.Adam(learning_rate = LEARNING_RATE)
	cur_cost = 1e12+0.1
	cur_img = None

	for i in range(iter):
		with tf.GradientTape() as tp:
			J_c = content_cost(c_img, gen_img)
			J_s = style_cost(s_img, gen_img)
			J_t = wc * J_c + ws * J_s

		grads = tp.gradient(J_t, gen_img)
		optimizer.apply_gradients([(grads, gen_img)])

		if J_t < cur_cost:
			cur_cost = J_t
			cur_img = gen_img.numpy()


		if i % int(iter/10) == 0:
			#print('Cost at {}: {}'.format(i, J_t))
			st.write(('Cost at {}: {}'.format(i, J_t)))
			image_gallery.append(gen_img.numpy())

	st.markdown("### Generated Image :")

	return cur_img

