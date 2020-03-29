# WSI-Segmentation-Patch

### Requirements

```
pip install -r requirements.txt
```

### Step 1: Extracting patches

```
python extract_patches.py -i <IMAGES_FOLDER> \ # <-- required
                          -o <OUTPUT_FOLDER> \ # <-- required
                          -ps <PATCH_SIZE> \ # <-- default: 256
                          -ppi <PATCHES_PER_IMAGE> \ # <-- default: 1000
                          -p <TRAIN_PROPORTION> \ # <-- default: 0.9
```

### Step 2: Run training

```
python train.py -ti <TRAIN_IMAGES_FOLDER> \ # <-- required
                -vi <VAL_IMAGES_FOLDER> \ # <-- required
                -m <MODEL_PATH> \ # <-- required
                -ps <PATCH_SIZE> \ # <-- default: 256
                -bs <BATCH_SIZE> \ # <-- default: 20
                -e <NB_EPOCHS> \ # <-- default: 10.000
```

### Step 3: Run inference

Run on a single Whole Slide Image:

```
python inference.py -i <IMAGE_PATH> \ # <-- required
                    -m <MODEL_PATH> \ # <-- required
                    -o <OUTPUT_PATH> \ # <-- required
                    -ps <PATCH_SIZE> \ # <-- default: 256
```

Run on all WSIs:

```
python inference_all.py -i <IMAGES_FOLDER> \ # <-- required
                        -m <MODEL_PATH> \ # <-- required
                        -o <OUTPUT_FOLDER> \ # <-- required
                        -ps <PATCH_SIZE> \ # <-- default: 256
```
