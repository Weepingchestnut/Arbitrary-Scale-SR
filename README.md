# This repo is built for survey paper: Arbitrary-Scale Super-Resolution via Deep Learning: A Comprehensive Survey

Arbitrary-scale image super-resolution (ISR) and video super-resolution (VSR) methods



## Scale-based taxonomy

![](./imgs/fig_scale_taxonomy.png)



##  Upsampling-based taxonomy

![](./imgs/fig_up_taxonomy.png)



Timeline of the development of deep learning-based arbitrary-scale super-resolution methods

![](./imgs/fig_time1.jpg)



### Interpolation Arbitrary-Scale Upsampling

<img src="./imgs/fig_pre-interpolation_ASISR.png" style="zoom:30%;" />

<img src="./imgs/fig_post-interpolation_ASISR.png" style="zoom:30%;" />

| Paper                                                        | Model | Code                                                         | Published                                                    |
| ------------------------------------------------------------ | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Accurate Image Super-Resolution Using Very Deep Convolutional Networks | VDSR  | [MATLAB](https://cv.snu.ac.kr/research/VDSR/VDSR_code.zip), [PyTorch](https://github.com/Lornatang/VDSR-PyTorch) | [CVPR'2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Kim_Accurate_Image_Super-Resolution_CVPR_2016_paper.html), [arXiv'2015](https://arxiv.org/abs/1511.04587) |
|                                                              |       |                                                              |                                                              |
|                                                              |       |                                                              |                                                              |
|                                                              |       |                                                              |                                                              |



### Learnable Adaptive Arbitrary-Scale Upsampling

#### Meta Upsampling

![](./imgs/fig_metasr.png)

| Paper                                                        | Model   | Code                                                   | Published                                                    |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------ | ------------------------------------------------------------ |
| Meta-SR: A Magnification-Arbitrary Network for Super-Resolution | Meta-SR | [PyTorch](https://github.com/XuecaiHu/Meta-SR-Pytorch) | [CVPR'2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Hu_Meta-SR_A_Magnification-Arbitrary_Network_for_Super-Resolution_CVPR_2019_paper.html) |
|                                                              |         |                                                        |                                                              |
|                                                              |         |                                                        |                                                              |



#### Adaptive Upsampling

![](./imgs/fig_arbsr.png)

| Paper                                                        | Model | Code                                                         | Published                                                    |
| ------------------------------------------------------------ | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Learning A Single Network for Scale-Arbitrary Super-Resolution | ArbSR | [PyTorch](https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR) | [ICCV'2021](https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Learning_a_Single_Network_for_Scale-Arbitrary_Super-Resolution_ICCV_2021_paper.html) |
|                                                              |       |                                                              |                                                              |
|                                                              |       |                                                              |                                                              |



### Implicit Neural Representation based Arbitrary-Scale Upsampling

![](./imgs/fig_liif_network.png)

| Paper                                                        | Model | Code                                      | Published                                                    |
| ------------------------------------------------------------ | ----- | ----------------------------------------- | ------------------------------------------------------------ |
| Learning Continuous Image Representation with Local Implicit Image Function | LIIF  | [PyTorch](https://github.com/yinboc/liif) | [CVPR'2021](https://openaccess.thecvf.com/content/CVPR2021/html/Chen_Learning_Continuous_Image_Representation_With_Local_Implicit_Image_Function_CVPR_2021_paper.html) |



![](./imgs/fig_liif_class.png)

| Paper                                                        | Model   | Code                                                         | Published                                      |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ | ---------------------------------------------- |
| **Spectral bias**                                            |         |                                                              |                                                |
| UltraSR: Spatial Encoding is a Missing Key for Implicit Image Function-based Arbitrary-Scale Super-Resolution | UltraSR | [PyTorch](https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR) | [arXiv'2021](https://arxiv.org/abs/2103.12716) |
|                                                              |         |                                                              |                                                |
|                                                              |         |                                                              |                                                |



### Other Arbitrary Scale Upsampling

| Paper                                                        | Model | Code                                                         | Published                                                    |
| ------------------------------------------------------------ | ----- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ASDN: A Deep Convolutional Network for Arbitrary Scale Image Super-Resolution | ASDN  | [PyTorch](https://github.com/alessandrodicosola/SuperSampling) | [Mob. Netw. Appl.'2021](https://link.springer.com/article/10.1007/s11036-020-01720-2) |
|                                                              |       |                                                              |                                                              |
|                                                              |       |                                                              |                                                              |
