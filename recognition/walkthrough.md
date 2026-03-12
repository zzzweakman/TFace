## 改动说明

本次改动为识别人脸训练新增了 `image_list` 数据模式，可直接使用 `jpg/png` 图片列表进行训练。

### 关键改动
- `torchkit/data/parser.py`
  - `IndexParser` 支持空格分隔与 `tab` 分隔。
  - `ImgSampleParser` 增加读图失败的显式异常。
  - `TFRecordSampleParser` 改为延迟导入 `dareblopy`。
- `torchkit/data/dataset.py`
  - 新增 `MultiImageListDataset`，用于读取 `path label` 格式的图片列表。
- `torchkit/data/sampler.py`
  - 新增 `ImageListDistributedSampler`。
- `torchkit/task/base_task.py`
  - 增加 `DATASET_MODE=image_list` 分支接入新 dataset/sampler。
- `train.yaml`
  - 指向 `CASIA_namelist.txt` 与 `/nfs/zzzhong/codes/exp/TFace/dataset/images` 对应的数据根目录。

### 验证结果
已完成 dataset/sampler/dataloader 冒烟验证（单进程模拟分布式环境）：

关键输出如下：
- `sample_num_on_rank0=452740`
- `total_sample_num=452740`
- `class_num=10572`
- `images_shape=(8, 3, 112, 112)`
- `labels_shape=(8,)`

说明：
- 当前运行环境缺少 `opencv-python`，无法做真实解码冒烟；本次验证使用最小 mock 验证了 `image_list` 模式的数据与采样链路。
- 真实训练前请先安装 `requirements.txt` 中依赖。
