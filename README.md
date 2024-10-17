# Microplastics_UNet: Automatic quantification and classification of microplastics in scanning electron micrographs via deep learning. 

This repository shared the source code for the paper *"Automatic quantification and classification of microplastics in scanning electron micrographs via deep learning"* published in the journal 
Science of the Total Environment. Four models were adapted for microplastics semantic segmentation, instance segmentation and shape classification.  Codes were adapted from the following repositories:

- U-Net: [https://github.com/zhixuhao/unet](https://github.com/zhixuhao/unet) semantic segmentation
- MultiResUNet: [https://github.com/nibtehaz/MultiResUNet](https://github.com/nibtehaz/MultiResUNet) semantic segmentation
- VGG-16: [https://github.com/sajadn/Exemplar-VAE](https://github.com/sajadn/Exemplar-VAE) shape classification
- Pixel-embedding UNet: [https://github.com/looooongChen/instance_segmentation_with_pixel_embeddings](https://github.com/looooongChen/instance_segmentation_with_pixel_embeddings) instance segmentation


  ## Data

A manually labelled SEM dataset of microplastics was built. Image segmentation and shape classification were performed on 3 classes: fibres,
beads and fragments as shown in the paper [https://doi.org/10.1016/j.scitotenv.2022.153903](https://doi.org/10.1016/j.scitotenv.2022.153903). Datasets were uploaded to Mendeley Data [https://data.mendeley.com/datasets/z6459vntbr/1](https://data.mendeley.com/datasets/z6459vntbr/1) (will soon be updated to https://data.mendeley.com/datasets/z6459vntbr/2)



  ## Citation

Cite our paper using the following bibtex item:
```
@article{shi2022automatic,
  title={Automatic quantification and classification of microplastics in scanning electron micrographs via deep learning},
  author={Shi, Bin and Patel, Medhavi and Yu, Dian and Yan, Jihui and Li, Zhengyu and Petriw, David and Pruyn, Thomas and Smyth, Kelsey and Passeport, Elodie and Miller, RJ Dwayne and others},
  journal={Science of The Total Environment},
  volume={825},
  pages={153903},
  year={2022},
  publisher={Elsevier}
}
```
Or: <br>
```
Shi, Bin, et al. "Automatic quantification and classification of microplastics in scanning electron micrographs via deep learning." Science of The Total Environment 825 (2022): 153903
```


## Acknowledgements

The project is supported by the WaterSeed Fund from Institute for Water Innovation, University of Toronto and by the Natural Sciences and Engineering Research Council of Canada (NSERC)'s Discovery Grant. Electron microscopy was performed at the Open Centre for the Characterization of Advanced Materials (OCCAM), funded by the Canada Foundation for Innovation. This research was enabled in part by support provided by Compute Canada.
