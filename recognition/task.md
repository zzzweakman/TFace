## 任务清单

- [x] 梳理当前 `local_train.sh` 与 `train.py` 的数据加载链路
- [x] 设计并实现支持 `jpg/png` 的 dataset 与 sampler
- [x] 接入 `BaseTask.make_inputs`，保证可通过配置切换
- [x] 更新 `train.yaml`，对接 `CASIA_namelist.txt` 与 `images` 数据
- [x] 完成 dataloader 冒烟验证并记录结果
