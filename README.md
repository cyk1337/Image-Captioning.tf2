# Image-Captioning

# Requirement
- Tensorflow >= 2.0
```bash
conda env create -f environment.yml
conda activate tf2_env
```
# Run
```bash
(tf2_env) $ python distributed_train.py --dataset COCO --do_train True --max_seq_len 30
```