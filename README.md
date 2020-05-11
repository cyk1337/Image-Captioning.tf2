# Image-Captioning

# Requirement
- Tensorflow >= 2.0
```bash
conda env create -f environment.yml
conda activate tf2_env
```

# Data
- data/
    - COCO/
        - img_fts
        - ...

# Run
```bash
(tf2_env) $ python main.py --dataset COCO \
                           --raw_data_path $DATA_PATH \
                           --do_train True \
                           --max_seq_len 30
                          
```