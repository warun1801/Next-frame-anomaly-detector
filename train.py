from losses import initialize_flownet
from model import Pix2Pix
from utils import get_train_test_files, get_data_gen, get_data_gen_frames
from keras.utils.vis_utils import plot_model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras
# params
FLOW_CHECKPOINT = 'checkpoints/pretrains/flownet-SD.ckpt-0'

batch_size = 5
timesteps = 5
im_width, im_height = 256, 256
# end params

initialize_flownet(sess, FLOW_CHECKPOINT)
# tf saver


train_files, test_files = get_train_test_files()
train_gen = get_data_gen_frames(files=train_files, timesteps=timesteps, batch_size=batch_size, im_size=(im_width, im_height))
print(train_gen)
gan = Pix2Pix(im_height=im_height, im_width=im_width, lookback=timesteps-1)
print("Generator Summary")
gan.generator.summary()
plot_model(gan.generator, to_file='gen_plot.png', show_shapes=True, show_layer_names=True)
print()
print("Discriminator Summary")
gan.discriminator.summary()
plot_model(gan.discriminator, to_file='dis_plot.png', show_shapes=True, show_layer_names=True)
print()
print("Combined Summary")
gan.combined.summary()
plot_model(gan.combined, to_file='gan_plot.png', show_shapes=True, show_layer_names=True)

gan.train(train_gen, epochs=600, batch_size=batch_size, save_interval=200, save_file_name="pedflow2_gen_t5.model")