This is a simple implementation of our paper accepted by TNNLS: Cross-Cloud Consistency for Weakly Supervised Point Cloud Semantic Segmentation

### (1) Setup
This code has been tested with Python 3.6, Tensorflow 1.14, and CUDA 10.0  on Ubuntu 16.04.

- Setup python environment
pip install -r helper_requirements.txt
sh compile_op.sh

### (2) Weakly supervised semantic segmentation on S3DIS
S3DIS dataset can be found 
<a href="https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1">here</a>. 
Download the files named "Stanford3dDataset_v1.2_Aligned_Version.zip". Uncompress the folder and move it to 
`Your_data_path/S3DIS`.

- Preparing the dataset:
```
python utils/data_prepare_s3dis.py
```
- Start Pretrain, Train, and Test on Area 5 of S3DIS dataset:
```
sh train_test_s3dis.sh
```

Citation:
If you find our approach useful in your research, please consider citing:

```
@article{zhang2022,
  author={Zhang, Yachao; Lan, Yuxiang; Xie, Yuan; Li, Cuihua; Qu, Yanyun},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Cross-Cloud Consistency for Weakly Supervised Point Cloud Semantic Segmentation}, 
  year={2024},
  volume={},
  number={},
  pages={1-13}}
```
