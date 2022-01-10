import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from moviepy.editor import CompositeVideoClip, ImageSequenceClip
from utils import get_data_gen_frames, get_test_files, get_train_test_files, denormalize
import matplotlib.pyplot as plt
import numpy as np
from losses import gradient_loss, intensity_loss, optical_flow_loss_farneback
from math import log10, sqrt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# Thresholds for the differences
I_THRESHOLD = 0.01
G_THRESHOLD = 0.001
F_THRESHOLD = 9 * 10e-7

FLOW_LOSS_WT = 10**7
INTESITY_LOSS_WT = 10
GRADIENT_LOSS_WT = 1

# params

batch_size = 1
timesteps = 5
im_width = im_height = 256
# end params

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def anomaly_ensemble(error_values):
    i_loss, g_loss, f_loss, psnr = error_values
    if i_loss > I_THRESHOLD or g_loss > G_THRESHOLD or f_loss > F_THRESHOLD or psnr < 30:
        return True
    return False

def generate_video(saved_model_path, video_category=None):
    model = load_model(saved_model_path, compile=False)

    which_one = video_category
    test_files = get_test_files()
    test_gen = get_data_gen_frames(files=test_files, timesteps=timesteps, batch_size=batch_size, im_size=(im_width, im_height))

    y_true = []
    y_pred = []

    for i in range(200):
        x, y = next(test_gen)
        y_true.extend(y)

        # print(y)
        predictions = model.predict_on_batch(x)
        # print(predictions)
        # print(np.mean(y-predictions))
        i_loss = intensity_loss(y, predictions, 1) * INTESITY_LOSS_WT
        g_loss = gradient_loss(y, predictions, 1) * GRADIENT_LOSS_WT
        f_loss = optical_flow_loss_farneback(x, y, predictions) * FLOW_LOSS_WT
        psnr = PSNR(denormalize(y), denormalize(predictions))
        print(f"{i+1} Intensity loss: {i_loss}, Gradient loss: {g_loss}, Flow loss: {f_loss}, PSNR: {psnr}")
        y_pred.extend(predictions)

        x, y, predictions = [], [], []


    clip1 = ImageSequenceClip([denormalize(i) for i in y_true], fps=5)
    clip2 = ImageSequenceClip([denormalize(i)for i in y_pred], fps=5)
    clip2 = clip2.set_position((clip1.w, 0))
    video = CompositeVideoClip((clip1, clip2), size=(clip1.w * 2, clip1.h))
    video.write_videofile("{}.mp4".format(which_one if which_one else "render"), fps=5)


def evaluate_model(saved_model_path):
    model = load_model(saved_model_path, compile=False)
    test_files = get_test_files()
    test_gen = get_data_gen_frames(files=test_files, timesteps=timesteps, batch_size=batch_size, im_size=(im_width, im_height))


    for i in range(70):
        x, y = next(test_gen)
        # y_true.extend(y)

        # print(y)
        predictions = model.predict_on_batch(x)
        # print(predictions)
        # print(np.mean(y-predictions))
        i_loss = intensity_loss(y, predictions, 2)
        g_loss = gradient_loss(y, predictions, 2)
        f_loss = optical_flow_loss_farneback(x, y, predictions)
        psnr = PSNR(denormalize(y), denormalize(predictions))
        print(f"{i+1} Intensity loss: {i_loss}, Gradient loss: {g_loss}, Flow loss: {f_loss}, PSNR: {psnr}")
        # y_pred.extend(predictions)
        if anomaly_ensemble((i_loss, g_loss, f_loss, psnr)):
            print("Anomaly Detected! Please check it out!!!!")
        x, y, predictions = [], [], []


def plot_different_models(timesteps = [5, 10]):
    """
    Compares ssim/psnr of different models. The models for each of the supplied timestap
    must be present
    param timesteps A list of numbers indicating the timesteps that were used for training different models
    """
    from skimage.measure import compare_psnr, compare_ssim
    psnrs = {}
    ssims = {}
    for ts in timesteps:
        model_name = "r_p2p_gen_t{}.model".format(ts)
        model = load_model(model_name)
        train_files, test_files = get_train_test_files()
        test_gen = get_data_gen_frames(files=train_files, timesteps=ts, batch_size=batch_size, im_size=(im_width, im_height))

        y_true = []
        y_pred = []

        for _ in range(200):
            x, y = next(test_gen)
            y_true.extend(y)

            predictions = model.predict_on_batch(x)
            y_pred.extend(predictions)
        psnrs[ts] = [compare_psnr(denormalize(yt), denormalize(p)) for yt, p in zip((y_true), (y_pred))]
        ssims[ts] = [compare_ssim(denormalize(yt), denormalize(p), multichannel=True) for yt, p in zip((y_true), (y_pred))]

    plt.boxplot([psnrs[ts] for ts in timesteps], labels=timesteps)
    plt.savefig("jigsaws_psnrs_all.png")

    plt.figure()
    plt.boxplot([ssims[ts] for ts in timesteps], labels=timesteps)
    plt.savefig("jigsaws_ssims_all.png")

# plot_different_models(timesteps=[5])
evaluate_model("ped_gan_convlstm_flow2_t5.model")