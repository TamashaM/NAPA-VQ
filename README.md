##NAPA-VQ: Neighborhood-Aware Prototype Augmentation with Vector Quantization for Continual Learning ##

#Requirements
Python 3.7.4
PyTorch: 1.9.0

#Datasets
Create the "dataset" directory under NAPA-VQ and download the following datasets into the created directory.

1.CIFAR-100

2.Tiny-ImageNet

3.ImageNet-Subset

#Training

CIFAR-100

``` 
# 5 Tasks
python main_cifar.py --fg_nc 50 --task_num 5 --custom_name "cifar_100-5" --base_model "cifar_100-5" --epochs 100 --shuffle > cifar-5.txt
# 10 Tasks
python main_cifar.py --fg_nc 50 --task_num 10 --custom_name "cifar_100-10" --base_model "cifar_100-10" --epochs 100 --shuffle > cifar-10.txt
# 20 Tasks
python main_cifar.py --fg_nc 40 --task_num 20 --custom_name "cifar_100-20" --base_model "cifar_100-20" --epochs 100 --shuffle > cifar-20.txt

```

Tiny-ImageNet
```
# 5 Tasks
python main_tiny.py --fg_nc 100 --task_num 5 --base_model "tiny-5" --custom_name "tiny-5" --shuffle --epochs 50 > tiny-5.txt
# 10 Tasks
python main_tiny.py --fg_nc 100 --task_num 10 --base_model "tiny-10" --custom_name "tiny-10" --shuffle --epochs 50 > tiny-10.txt
# 20 Tasks
python main_tiny.py --fg_nc 100 --task_num 20 --base_model "tiny-20" --custom_name "tiny-20" --shuffle --epochs 50 > tiny-20.txt
```

ImageNet-Subset

```
# 5 Tasks
python main_imagenet.py --custom_name "imagenet-5" --base_model "imagenet-5" --fg_nc 50 --task_num 5 --shuffle > imagenet-5.txt
# 10 Tasks
python main_imagenet.py --custom_name "imagenet-10" --base_model "imagenet-10" --fg_nc 50 --task_num 10 --shuffle > imagenet-10.txt
# 20 Tasks
python main_imagenet.py --custom_name "imagenet-20" --base_model "imagenet-20" --fg_nc 40 --task_num 20 --shuffle > imagenet-20.txt

```

#References

We thank the authors of following repositories for their excellent codebase providing reusable functions.

https://github.com/Impression2805/CVPR21_PASS