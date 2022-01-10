import numpy as np
import tensorflow as tf
import cv2 as cv
from flownet2.src.flowlib import flow_to_image
from flownet2.src.flownet_sd.flownet_sd import FlowNetSD  # Ok
from flownet2.src.training_schedules import LONG_SCHEDULE
from flownet2.src.net import Mode
slim = tf.contrib.slim
FLOW_WIDTH = 256
FLOW_HEIGHT = 256
FLOW_CHECKPOINT = 'checkpoints/pretrains/flownet-SD.ckpt-0'

def flownet(input_a, input_b, height, width, reuse=None):
    input_a = input_a[:, -1, ...]
    net = FlowNetSD(mode=Mode.TEST)
    # net.load_weights(FLOW_CHECKPOINT)
    input_a = (input_a + 1.0) / 2.0     # flownet receives image with color space in [0, 1]
    input_b = (input_b + 1.0) / 2.0     # flownet receives image with color space in [0, 1]
    print(input_a.shape, input_b.shape)
    input_a = tf.image.resize_images(input_a, [height, width])
    input_b = tf.image.resize_images(input_b, [height, width])
    flows = net.model(
        inputs={'input_a': input_a, 'input_b': input_b},
        training_schedule=LONG_SCHEDULE,
        trainable=False, reuse=reuse
    )
    return flows['flow']


def initialize_flownet(sess, checkpoint):
    flownet_vars = slim.get_variables_to_restore(include=['FlowNetSD'])
    print(flownet_vars)
    flownet_saver = tf.train.import_meta_graph(checkpoint + '.meta')
    print('FlownetSD restore from {}!'.format(checkpoint))
    flownet_saver.restore(sess, checkpoint)

def optical_flow_loss(img_seq_A, img_B, fake_B):
    real_flow = flownet(img_seq_A, img_B, FLOW_HEIGHT, FLOW_WIDTH, reuse=None)
    fake_flow = flownet(img_seq_A, fake_B, FLOW_HEIGHT, FLOW_WIDTH, reuse=True)
    return tf.reduce_mean(tf.abs(real_flow - fake_flow))

def intensity_loss(real, gen, l_num):
    return np.mean(np.abs((real - gen) ** l_num))


def gradient_loss(real, gen, alpha):
    real = np.squeeze(real)
    gen = np.squeeze(gen)
    grad_real = np.array(np.gradient(real))
    grad_gen = np.array(np.gradient(gen))
    return np.mean(np.abs(grad_real - grad_gen) ** alpha)

def optical_flow_farneback(prev_frame, curr_frame):
    prev_frame = np.float32(prev_frame)
    curr_frame = np.float32(curr_frame)
    mask = np.zeros_like(prev_frame)
    mask[..., 1] = 1.0
    prev_frame = cv.cvtColor(prev_frame, cv.COLOR_RGB2GRAY)
    curr_frame = cv.cvtColor(curr_frame, cv.COLOR_RGB2GRAY)
    flow = cv.calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def optical_flow_loss_farneback(img_seq_A, img_B, fake_B, alpha = 1):
    real_flow = optical_flow_farneback(img_seq_A[0,-1,...], img_B.squeeze())
    fake_flow = optical_flow_farneback(img_seq_A[0,-1,...], fake_B.squeeze())
    return np.mean(np.abs((real_flow - fake_flow)**alpha))