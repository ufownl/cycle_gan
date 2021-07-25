# cycle_gan

CycleGAN with Spectral Normalization and Class Activation Mapping Attention implemented using [MXNet](https://mxnet.incubator.apache.org/).

I tried to train it using the dataset [selfie2anime](https://www.kaggle.com/arnaud58/selfie2anime) and got the following results:

![p1](/docs/w1.jpg)
![p2](/docs/w2.jpg)
![p3](/docs/w3.jpg)
![p4](/docs/w4.jpg)
![p5](/docs/w5.jpg)
![p6](/docs/w6.jpg)
![p7](/docs/w7.jpg)
![p8](/docs/w8.jpg)
![p9](/docs/w9.jpg)

## Requirements

* [Python3](https://www.python.org/)
  * [MXNet](https://mxnet.apache.org/)
  * [GluonCV](https://gluon-cv.mxnet.io/)
  * [NumPy](https://www.numpy.org)
  * [opencv-python](https://github.com/skvark/opencv-python)
  * [Matplotlib](https://matplotlib.org/)
  * [PyPNG](https://github.com/drj11/pypng)
  * [Dlib](http://dlib.net/) (Only needed for selfie2anime)

## Usage

### Get pre-trained models

You can checkout the pre-trained models in pretrained/\* branches. For example, use the following command to get the pre-trained selfie2anime model: 

```
git clone --branch pretrained/selfie2anime https://github.com/ufownl/cycle_gan.git
```

### Run selfie2anime cli-demo

Simplest:

```
python3 selfie2anime.py /path/to/selfie
```

Details:

```
usage: selfie2anime.py [-h] [--resize RESIZE] [--device_id DEVICE_ID] [--gpu]
                       IMG [IMG ...]

Start a selfie2anime tester.

positional arguments:
  IMG                   path of the image file[s]

optional arguments:
  -h, --help            show this help message and exit
  --resize RESIZE       set the short size of fake image (default: 256)
  --device_id DEVICE_ID
                        select device that the model using (default: 0)
  --gpu                 using gpu acceleration
```

### Run selfie2anime demo server

In addition to the cli-demo above, you can also run the demo server of selfie2anime, then click [here](https://ufownl.github.io/cycle_gan/selfie2anime.html) to visit the demo page.

Simplest:

```
python3 server.py --model selfie2anime
```

Details:

```
usage: server.py [-h] [--reversed] [--model MODEL] [--resize RESIZE] [--addr ADDR]
                 [--port PORT] [--device_id DEVICE_ID] [--gpu]

This is CycleGAN demo server.

optional arguments:
  -h, --help            show this help message and exit
  --reversed            reverse transformation
  --model MODEL         set the model used by the server (default: vangogh2photo)
  --resize RESIZE       set the short size of fake image (default: 256)
  --addr ADDR           set address of cycle_gan server (default: 0.0.0.0)
  --port PORT           set port of cycle_gan server (default: 80)
  --device_id DEVICE_ID
                        select device that the model using (default: 0)
  --gpu                 using gpu acceleration
```

## References

* [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://junyanz.github.io/CycleGAN/)
* [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)
* [U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation](https://arxiv.org/abs/1907.10830)
