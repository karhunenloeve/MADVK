# Deep Learning Estimation of Absorbed Dose for Nuclear Medicine Diagnostics

To calculate the absorbed radiation dose in dosimetry, Monte Carlo simulations are performed on tissue densities from a CT imaging to calculate the deposited energy. The transfer of mass densities into the so-called dose-voxel-kernels corresponds to an image-to-image transfer. This transfer can be learned from image reconstruction and image segmentation techniques. Simulating dose-voxel kernels is extremely time-consuming and requires large resources. In the following, the algorithm is presented, which provides an alternative and fast solution for the calculation of DVKs. Sections are taken from CT imaging, which are called mass kernels. The entire CT image can be decomposed into these mass kernels. Then the Monte-Carlo simulation is performed on the mass kernel. The radiation energy of an isotope in the center of the mass nucleus is simulated. The result is the dose-voxel nuclei. These are then used for convolution of a SPECT imaging to obtain the distribution of the absorbed radiation dose. The transfer from mass nuclei to dose-voxel nuclei is replaced by the neural network.

**Keywords:** Deep learning · Dosimetry · Artificial intelligence · Nuclear medicine · Cancer therapy.

## Citation

[Luciano Melodia (2018): Deep Learning Estimation of Absorbed Dose for Nuclear Medicine Diagnostics. University of Regensburg, Master Thesis.](https://arxiv.org/abs/1805.09108).

    @thesis{MADVK2018,
     author    = "Luciano Melodia",
     title     = "Deep Learning Estimation of Absorbed Dose for Nuclear Medicine Diagnostics",
     year      = "2018",
     month     = "April",
     pages     = "1-36",
     publisher = "University of Regensburg",
     url       = "https://arxiv.org/abs/1805.09108",
    }
    
## Requirements

**Versions:** ```matplotlib 3.5.1```, ```keras 2.8.0```, ```numpy 1.22.3```, ```Pillow 9.0.1```, ```scipy 1.8.0```, ```tensorflow 2.8.0```, ```scikit-learn 0.0.20```

* [Keras](https://keras.io/) - Keras Tensorflow Wrapper.
* [Scikit-Learn](http://scikit-learn.org/stable/index.html/) – some preprocessing.
* [Tensorflow](https://www.tensorflow.org/) – Tensorflow as backend.
