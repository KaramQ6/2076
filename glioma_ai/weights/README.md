# Model Weights Management

Pre-trained model weights for the 3D Glioma SegResNet architecture:
- **Architecture:** MONAI SegResNet 3D (`init_filters=16`, `blocks_down=[1, 2, 2, 4]`, `blocks_up=[1, 1, 1]`)
- **Benchmark Performance:** WT Dice: 0.9171, TC Dice: 0.9103, ET Dice: 0.8625 (Tested on 16 held-out BraTS-TCGA-GBM subjects)
- **Local File:** `best_segresnet_weights.pth` (18.5 MB)
- **Release Download:** Available as a release asset under [GitHub Releases v1.0.0](https://github.com/KaramQ6/2076/releases).

If `best_segresnet_weights.pth` is not present locally, the pipeline can automatically download the official MONAI bundle:
```bash
python -m monai.bundle download --name brats_mri_segmentation --bundle_dir C:/tmp/glioma_scratch
```
