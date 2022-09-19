# The Close Relationship Between Contrastive Learning and Meta-Learning

### Abstract
Contrastive learning has recently taken off as a paradigm for learning from unlabeled data. In this paper, we discuss the close relationship between contrastive learning and meta-learning under a certain task distribution. We complement this observation by showing that established meta-learning methods, such as Prototypical Networks, achieve comparable performance to SimCLR when paired with this task distribution. This relationship can be leveraged by taking established techniques from meta-learning, such as task-based data augmentation, and showing that they benefit contrastive learning as well. These tricks also benefit state-of-the-art self-supervised learners without using negative pairs such as BYOL, which achieves 94.6\% accuracy on CIFAR-10 using a self-supervised ResNet-18 feature extractor trained with our meta-learning tricks.  We conclude that existing advances designed for contrastive learning or meta-learning can be exploited to benefit the other, and it is better for contrastive learning researchers to take lessons from the meta-learning literature (and vice-versa) than to reinvent the wheel.


## SSL training Examples

1. To train with SimCLR for 100 epochs on ImageNet with batch size 256:

    ```bash
    python train.py --config configs/imagenet_train_epochs100_bs512.yaml --dist_address '127.0.0.1:1672' --n_supp 1 --n_query 1 --multiplier 2 --arch ResNet50 --seed 1234 --run_id 000001 --head contrastive --accu_iter 1 --batch_size 256 --lr 2.4 --iters 1000800 --eval_freq 100080 --save_freq 12510 --temperature 10.
    ```

3. To train with SimCLR with R2D2 head for 100 epochs on ImageNet with batch size 256:
    ```bash
    python train.py --config configs/imagenet_train_epochs100_bs512.yaml --dist_address '127.0.0.1:1672' --n_supp 1 --n_query 1 --multiplier 2 --arch ResNet50 --seed 1234 --run_id 000001 --head R2D2 --accu_iter 1 --batch_size 256 --lr 2.4 --iters 1000800 --eval_freq 100080 --save_freq 12510 --temperature 0.1
    ```
## Evaluation Examples
1. To test models for linear evaluation:
```
python train.py --config configs/imagenet_eval.yaml --encoder_ckpt logs/exman-configs/imagenet_train_epochs100_bs512.yaml/runs/000001/checkpoint-1000800.pth.tar --run_id 000001
```
2.  To test models for semi-supervised learning with 1% labeled data:
```
python train.py --config configs/imagenet_semi_eval.yaml --train_size 1 --encoder_ckpt logs/exman-configs/imagenet_train_epochs100_bs512.yaml/runs/000005/checkpoint-1000800.pth.tar --eval_freq 100 --log_freq 100 --run_id 000001 --iters 3000 --lr 0.05
```
3. To test models for semi-supervised learning with 10% labeled data:
```
python train.py --config configs/imagenet_semi_eval.yaml --encoder_ckpt logs/exman-configs/imagenet_train_epochs100_bs512.yaml/runs/000001/checkpoint-1000800.pth.tar --eval_freq 100 --log_freq 100 --run_id 000001 --iters 60000 --train_size 10 --lr 0.1
```
## Acknowledgments

This code is based on the implementations of [**SimCLR**]https://github.com/AndrewAtanov/simclr-pytorch, [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet) and [**MetaAug**](https://github.com/RenkunNi/MetaAug).
