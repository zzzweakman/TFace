## 实施方案

### 目标
在不破坏原有 tfrecord 训练链路的前提下，新增可读取 `jpg/jpeg/png` 的训练数据路径。

### 步骤
1. 在 `torchkit/data/parser.py` 增强索引解析：兼容 `tab` 与空格分隔的 `path label`。
2. 在 `torchkit/data/dataset.py` 新增 `MultiImageListDataset`：
   - 输入：`INDEX_ROOT/<name>.txt`，每行为 `relative_path label`。
   - 输出：`(image_tensor, label)`。
   - 支持扩展名过滤（默认 `jpg/jpeg/png`）。
3. 在 `torchkit/data/sampler.py` 新增 `ImageListDistributedSampler`，复用分布式采样语义。
4. 在 `torchkit/task/base_task.py` 增加 `DATASET_MODE=image_list` 分支。
5. 更新 `train.yaml` 到 `CASIA_namelist.txt` 与本地 `images` 路径。

### 风险与回滚
- 风险：列表文件格式异常（空行、错误分隔符、坏图）。
- 缓解：加入空行过滤和读图失败显式报错。
- 回滚：将 `DATASET_MODE` 改回 `auto` 并恢复原 `DATA_ROOT/INDEX_ROOT/DATASETS`。

### 验证方式
1. 仅初始化 dataset/sampler/dataloader，读取首个 batch。
2. 检查 batch 的 tensor shape、label 范围、样本路径可读性。
