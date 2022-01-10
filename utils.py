import os
import glob
import random
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip

data_dir = "ped2\\training\\frames"
test_dir = "ped2\\testing\\frames"

def get_train_test_files(split_ratio=0.8, shuffle=True, which=None):
    files = []
    # for video_dir in video_dirs:
    #     # vid_dir_path = os.path.join(data_dir, video_dir, "video")
    video_files = glob.glob(os.path.join(data_dir, '*'))
    files.extend(video_files)
    print("Files: ",files)
    # sort files
    files = sorted(files)
    if shuffle:
        random.shuffle(files)

    total_files = len(files)
    len_train_files = int(total_files * split_ratio)
    return files[:len_train_files], files[len_train_files:]

def get_test_files():
    files = []
    video_files = glob.glob(os.path.join(test_dir, '*'))

    return video_files

def normalize_image(img):
    return img / 255.0

def denormalize(img):
    return (img * 255).astype('uint8')

def _get_xy_pair(frames, timesteps, frame_mode, im_size):
    rng = range(timesteps, len(frames)-1, timesteps) if frame_mode=="unique" else range(timesteps, len(frames)-1)

    def resize_image(np_image):
        return np.array(Image.fromarray(np_image, mode="RGB").resize(im_size))
    for i in rng:
        x = frames[i-timesteps: i]
        x = list(map(resize_image, x))
        y = resize_image(frames[i+1])

        yield x, y


def get_data_gen(files, timesteps=5, fps=15, batch_size=32, frame_mode="unique", for_training=True, im_size=(198, 198)):
    x_batch = []
    y_batch = []
    while True:
        for file in files:
            clip = VideoFileClip(file, audio=False)
            frames = list(clip.iter_frames(fps=fps))
            clip.close()
            for x, y in _get_xy_pair(frames, timesteps=timesteps, frame_mode=frame_mode, im_size=im_size):
                x_batch.append(x)
                y_batch.append(y)
                if len(x_batch) >= batch_size:
                    yield normalize_image(np.array(x_batch)), normalize_image(np.array(y_batch))
                    x_batch = []
                    y_batch = []
        if not for_training:
            break

def get_data_gen_frames(files, timesteps=5, fps=15, batch_size=32, frame_mode="unique", for_training=True, im_size=(198, 198)):
    x_batch = []
    y_batch = []
    while True:
        # print("IN")
        # print(files)
        for file in files:
            print(file)
            frames = []
            for frm in os.listdir(file):
                frames.append(np.array(Image.open(os.path.join(file, frm))))
            # print("YEPPPPPP")
            for x, y in _get_xy_pair(frames, timesteps=timesteps, frame_mode=frame_mode, im_size=im_size):
                x_batch.append(x)
                y_batch.append(y)
                if len(x_batch) >= batch_size:
                    yield normalize_image(np.array(x_batch)), normalize_image(np.array(y_batch))
                    x_batch = []
                    y_batch = []
        if not for_training:
            break