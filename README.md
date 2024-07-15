# FlowAVSE: Efficient audio visual speech enhancement models with conditional flow matching



![Alt text](comparison.pdf)

FlowAVSE is an efficient audio-visual speech enhancement model. 
We compare it to recent speech enhancement models that use a diffusion approach. As illustrated in the image above, our model is approximately 22 times faster than AVDiffuSS and other models. [AVDiffuSS](https://mmai.io/projects/avdiffuss/) is a speech separation model that utilizes a diffusion method. To directly compare diffusion and flow matching methods, our model's architecture is similar to that of AVDiffuSS, differing only in the use of flow matching instead of diffusion.



This repository contains the official PyTorch implementation for the paper:

- [FlowAVSE: Efficient audio visual speech enhancement models with conditional flow matching ](https://arxiv.org/abs/2406.09286)

Our demo page is [here](https://cyongong.github.io/FlowAVSE.github.io/).

## Installation

- Create a new virtual environment with the following command. You should change the environment path in yaml file.
- `conda create -n FlowAVSE python=3.8`
- `pip install -r requirements.txt`

## Pre-trained checkpoints

- We provide pre-trained checkpoint for the model trained for 15 epochs on the VoxCeleb2 train dataset. It can be used for testing on both the VoxCeleb2 and LRS3 test datasets. The file can be downloaded [here](https://drive.google.com/file/d/1pwRD6Qhr1JEV7nNX0HE8HoMvUGGwQkvS/view?usp=sharing).

Usage:
- For evaluating the pre-trained checkpoint, use the `--testset` option of `test_se.py` (see section **Evaluation** below) for selecting the test dataset among VoxCeleb2 and LRS3. Use `--ckpt` option to specify the path of the checkpoint for `test_se.py`.

## Training

For training, run
```bash
python train.py 
```
It you don't want to save checkpoints, add --nolog option.

## Evaluation

To evaluate on a test set, run
```bash
python test_se.py --testset <'vox' or 'lrs3'> --ckpt /path/to/model/checkpoint --data_dir /path/to/test/data/directory
```
Use 'vox' for VoxCeleb2 test set, and 'lrs3' for LRS3 test set. You can obtain scores quickly because the training file only utilizes the initial 2.04 seconds per audio for inference.

If you wish to evaluate the entire audio, please run
```bash
python test_whole.py --testset <'vox' or 'lrs3'> --ckpt /path/to/model/checkpoint --data_dir /path/to/test/data/directory
```


The performance of the provided checkpoint evaluated by the first test command is as follows:

| testset | PESQ|ESTOI|SI-SDR|
|---------|-----|-----|------|
|VoxCeleb2|2.096|0.796|13.370|
|   LRS3  |2.077|0.850|14.353|


## Citation

```
@article{jung2024flowavse,
  title={FlowAVSE: Efficient Audio-Visual Speech Enhancement with Conditional Flow Matching},
  author={Jung, Chaeyoung and Lee, Suyeon and Kim, Ji-Hoon and Chung, Joon Son},
  journal={arXiv preprint arXiv:2406.09286},
  year={2024}
}
```