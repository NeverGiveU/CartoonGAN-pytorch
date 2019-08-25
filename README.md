## Description
This is an implementation of [CartoonGAN](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf). We provide both ".py" and "ipynb" version for tha training. The testing file will be uploaded soon.
## Reference
We got greate inspiration from [ComixGAN](https://github.com/nijuyr/comixGAN), and we used the implementation from this repository for blurring the comic images.
## How to Start?
### Firstly, prepare your datasets.
```python
─datasets
  ├─comic
  │  ├─001.jpg
  │  ├─002.jpg
  │  ├─...
  │  └─xxx.jpg
  ├─comic_blurred
  └─real
     ├─001.jpg
     ├─002.jpg
     ├─...
     └─xxx.jpg
```
### Secondly, blur the comic images.
```sh
python comic_blurring.py
```
### Attention
You need to delete the __tmp.txt__ in the three sub-dir in **datasets** before you save images in the corresponding directories.
### Train
run `python train3.py` to start training.
The Training consists of three parts, which is inspired by [ComixGAN](https://github.com/nijuyr/comixGAN), they are
```python
1) Pretrain the Generator to reconstruct the input clear comic images;
2) Pretrain the Discriminator to distinguish images among comic, blurred comic and real images;
** we use LSGAN instead.
3) Adversarially train both the Generator and Discriminator.
```
## Some Results
Note that because of the portraiture right of some real images, we just display the cartoonized results of some data.
![alt sample_1](https://github.com/NeverGiveU/CartoonGAN-pytorch/blob/master/sample_images/0015-001324.jpg)
![alt sample_1](https://github.com/NeverGiveU/CartoonGAN-pytorch/blob/master/sample_images/0017-001324.jpg)
![alt sample_1](https://github.com/NeverGiveU/CartoonGAN-pytorch/blob/master/sample_images/0024-001324.jpg)
![alt sample_1](https://github.com/NeverGiveU/CartoonGAN-pytorch/blob/master/sample_images/0026-001324.jpg)
![alt sample_1](https://github.com/NeverGiveU/CartoonGAN-pytorch/blob/master/sample_images/0037-001324.jpg)
![alt sample_1](https://github.com/NeverGiveU/CartoonGAN-pytorch/blob/master/sample_images/0048-001324.jpg)
![alt sample_1](https://github.com/NeverGiveU/CartoonGAN-pytorch/blob/master/sample_images/0051-001324.jpg)
![alt sample_1](https://github.com/NeverGiveU/CartoonGAN-pytorch/blob/master/sample_images/0072-001324.jpg)
