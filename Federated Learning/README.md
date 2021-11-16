

python=3.6  
pytorch=1.4.0


## Dataset Preparation

Prepare the dataset in the following structure for easy use of the code.The provided data loaders is ready for this this format and you may change it as your need.

```bash


                   |-- T1 
                   |                       
                   |                |--xxx.h5  
Dataset Folder-----|      |--train--|...
                   |      |         |...
                   |      |
                   |      |         |--xxx.h5 
                   |-- T2-|-- val --|...  
                          |         |...
                          |
                          |         |--xxx.h5
                          |--test --|...
                                    |...
```
An example of preprocessing BraTS dataset can be found at <code> utils/preprocess_datasets_brats.py </code>.

## Links for downloading the public datasets:


1) BraTS Dataset - <a href="https://www.med.upenn.edu/cbica/brats2020/data.html"> Link </a> 



# Run

## train FL-CT-SCANS
```bash 
python fl_images.py --phase train --dataset mri --model unet --epochs 50 --challenge singlecoil --local_bs 16 --num_users 4 --local_ep 2 --train_dataset BFHI --test_dataset H --sequence T1  --accelerations 4 --center-fractions 0.08 --val_sample_rate 1.0 --save_dir 'Dir path for saving checkpoints' --verbose
```
## train FL-multi-images
```bash 
python fl_multi-images.py --phase train --dataset mri --model unet --epochs 50 --challenge singlecoil --local_bs 16 --num_users 4 --local_ep 2 --train_dataset BFHI --test_dataset B --sequence T1 --accelerations 4 --center-fractions 0.08 --val_sample_rate 1.0 --save_dir 'Dir path for saving checkpoints' --verbose
```
## monitor the traning process
```bash 
tensorboard --logdir 'Dir path for saving checkpoints'
```
## test
```bash 
python test.py --phase test --dataset mri --challenge singlecoil --local_bs 16 --model unet --test_dataset I --sequence T1 --accelerations 4 --center-fractions 0.08 --save_dir 'Dir path for saving result'  --checkpoint 'checkpoint path for testing'  --verbose
