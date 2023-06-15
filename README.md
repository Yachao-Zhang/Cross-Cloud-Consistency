This code is supplementary material for TCBY submission paper (C3).
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
