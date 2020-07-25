from train import train_nst, get_image_gallery
from processing_img import preprocess_image, show_image


def start_training(c_img, s_img, iters, W_C, W_S):
	c_img = preprocess_image(c_img)
	s_img = preprocess_image(s_img)
	result_img = train_nst(c_img, s_img, iters, W_C, W_S)
	return show_image(result_img)

