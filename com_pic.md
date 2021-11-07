+ 更改 *experiment_model* 以调换训练模型（resnet18、VIT、resnet18_VIT）

+ 打开 *experiment/对应模型* 中的config文件可以查看参数设置

+ 运行代码时请将数据集放在data文件夹内，如下所示：

  data

  ​	images

  ​	mask

  ​	test_data.csv

  ​	train_data.csv

+ 最好结果：

  |                   | resnet18 |  VIT  | resnet18+VIT | resnet18(pretrained=False) |
  | :---------------: | :------: | :---: | :----------: | :------------------------: |
  |        f1         |  0.792   | 0.506 |    0.787     |           0.614            |
  | balanced accuracy |  0.844   | 0.649 |    0.846     |           0.720            |

  