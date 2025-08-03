[Dataset Download](https://drive.google.com/file/d/1CuMuKmBo89M5g8yHXmRtWQsNbd1GEBB2/view?usp=drive_link)

## Dataset Structure

Dataset is organised like this:

```
train_unzipped/
├── train/
│   ├── Fe/
│   └── X-ray/
├── val/
│   ├── Fe/
│   └── X-ray/
└── test/
    ├── Fe/
    └── X-ray/
```

Each class folder contains `.tif` images.  
This structure works seamlessly with `torchvision.datasets.ImageFolder`.
