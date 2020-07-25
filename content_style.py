import tensorflow as tf
from get_model import get_model
from tensorflow.keras import Model

############################################################################
loaded_model, model_summary = get_model("VGG19")

style_layers = [
    'block1_conv1', 
    'block3_conv1', 
    'block5_conv1'
]

content_layer = 'block5_conv2'

content_model = Model(
    inputs = loaded_model.input, 
    outputs = loaded_model.get_layer(content_layer).output
)

style_models = [Model(inputs = loaded_model.input, outputs = loaded_model.get_layer(layer).output) for layer in style_layers]

style_weight = 1. / len(style_models)
############################################################################

def content_cost(content, generated):
    a_C = content_model(content)
    a_G = content_model(generated)
    cost = tf.reduce_mean(tf.square(a_C - a_G))
    return cost

############################################################################
def gram_matrix(A):
    channels = int(A.shape[-1])
    a = tf.reshape(A, [-1, channels]) 
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a = True)
    return gram / tf.cast(n, tf.float32)

############################################################################
def style_cost(style, generated):
    J_s = 0
    
    for sm in style_models:
        a_S = sm(style)
        a_G = sm(generated)
        GS = gram_matrix(a_S)
        GG = gram_matrix(a_G)
        cur_cost = tf.reduce_mean(tf.square(GS - GG))
        J_s += cur_cost * style_weight
    return J_s