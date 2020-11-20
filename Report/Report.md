# Report

## 测试数据生成

### flip_lr

### rotate_r

### bright

### gaussian

### crop

### mixUp

## 测试数据评估

### CIFAR-10

|                        | original | flip_lr | rotate_r | bright | gaussian | crop | mixUp |
| ---------------------- | -------- | ------- | -------- | ------ | -------- | ---- | ----- |
| CNN_with_dropout       | 0.642    |         |          |        |          |      |       |
| CNN_without_dropout    | 0.739    |         |          |        |          |      |       |
| lenet5_with_dropout    | 0.679    |         |          |        |          |      |       |
| lenet5_without_dropout | 0.746    |         |          |        |          |      |       |
| random1_cifar10        | 0.108    |         |          |        |          |      |       |
| random2_cifar10        | 0.120    |         |          |        |          |      |       |
| ResNet_v1              | 0.698    |         |          |        |          |      |       |
| ResNet_v2              | 0.707    |         |          |        |          |      |       |

### CIFAR-100

|                        | original | flip_lr | rotate_r | bright | gaussian | crop | mixUp |
| ---------------------- | -------- | ------- | -------- | ------ | -------- | ---- | ----- |
| CNN_with_dropout       | 0.421    |         |          |        |          |      |       |
| CNN_without_dropout    | 0.806    |         |          |        |          |      |       |
| lenet5_with_dropout    | 0.525    |         |          |        |          |      |       |
| lenet5_without_dropout | 0.904    |         |          |        |          |      |       |
| random1                | 0.457    |         |          |        |          |      |       |
| random2                | 0.326    |         |          |        |          |      |       |
| ResNet_v1              | 0.620    |         |          |        |          |      |       |
| ResNet_v2              | 0.733    |         |          |        |          |      |       |