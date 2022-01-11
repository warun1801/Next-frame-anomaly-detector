<br/>
<p align="center">
  

  <h3 align="center">Next Frame Anomaly Detector</h3>

  <p align="center">
    A novel static anomaly detector using predicted RGB video frames.
    <br/>
    <br/>
    <a href="https://github.com/warun1801/next-frame-anomaly-detector"><strong>Explore the docs »</strong></a>
    <br/>
    <br/>
    <a href="https://github.com/warun1801/next-frame-anomaly-detector">View Demo</a>
    .
    <a href="https://github.com/warun1801/next-frame-anomaly-detector/issues">Report Bug</a>
    .
    <a href="https://github.com/warun1801/next-frame-anomaly-detector/issues">Request Feature</a>
  </p>
</p>

![Downloads](https://img.shields.io/github/downloads/warun1801/next-frame-anomaly-detector/total) ![Contributors](https://img.shields.io/github/contributors/warun1801/next-frame-anomaly-detector?color=dark-green) ![Issues](https://img.shields.io/github/issues/warun1801/next-frame-anomaly-detector) ![License](https://img.shields.io/github/license/warun1801/next-frame-anomaly-detector) 

## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## About The Project

![Screen Shot](images/screenshot.png)

Next frame anomaly detector is a GAN (Generative Adversarial Network) based implementation of a video frame predictor for anomaly detection. It is mainly used for detecting static anomalies of various kinds. This is due to the various kinds of losses used in the model. 

Losses:

* Optical flow loss
* Adversarial loss
* Intensity loss
* Gradient loss

## Working

The main idea of this is comparing the next predicted (normal) frame with the actual anomalous frame and comparing and thresholding the losses. Each of the loss corresponds to a different kind of anomaly.

## Built With

Built in python. ML frameworks like Tensorflow and Keras were majorly used. An implementation of FlowNet2 was used for calculating the flow of the images.

## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

Get the checkpoints for running the flownet from this [Link](https://drive.google.com/drive/folders/1An-SCJ06zd94vAV1kphRw1h1Nl6itmYE?usp=sharing)
Download the entire folder and put it in the same directory.

In utils.py
```python
data_dir = "your_folder\\training\\frames"
test_dir = "your_folder\\testing\\frames"
```

In train.py
```python
gan.train(train_gen, epochs=600, batch_size=batch_size, save_interval=200, save_file_name="your_model_name.model")
```

To train:
```sh
python train.py
```
To test:
```sh
python utils.py
```
### Prerequisites

You will need python version 3.6 to run this, since we are using tensorflow 1.
All the requirements are there in requirements.txt

Just do the following to install them:
```sh
pip install -r requirements.txt
```

## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.


## Roadmap

See the [open issues](https://github.com/warun1801/next-frame-anomaly-detector/issues) for a list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com/warun1801/next-frame-anomaly-detector/issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please make sure you check your spelling and grammar.
* Create individual PR for each suggestion.
* Please also read through the [Code Of Conduct](https://github.com/warun1801/next-frame-anomaly-detector/blob/main/CODE_OF_CONDUCT.md) before posting your first idea as well.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [LICENSE](https://github.com/warun1801/next-frame-anomaly-detector/blob/main/LICENSE.md) for more information.

## Authors

* **Warun Panpaliya** - *VNIT Computer Science 2022* - [Warun Panpaliya](https://github.com/warun1801/)
* **Samruddhi Pande** - *VNIT Computer Science 2022* - [Samruddhi Pande](https://github.com/samupande/)
* **Dhruv Sharma** - *VNIT Computer Science 2022*
* **Revin Gohil** - *VNIT Computer Science 2022*
## Acknowledgements

