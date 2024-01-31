# Hierarchical Region-level Decoupling Knowledge Distillation for Semantic Segmentation
A pytorch implenment for Hierarchical Region-level Decoupling Knowledge Distillation for Semantic Segmentation



## Training and testing

-   Training on one GPU:

```pycon
python train.py {config}
```

-   Testing on one GPU:

```pycon
python test.py {config}
```
{config} means the config path. The config path can be found in [new_configs](configs "new_configs").

# Acknowledgement

Specially thanks to [MMSegmentation](https://github.com/open-mmlab/mmsegmentation "MMSegmentation"), [MMEngine](https://github.com/open-mmlab/mmengine "MMEngine") and [MMRazor](https://github.com/open-mmlab/mmrazor "MMRazor").

# Citation

```bash
@misc{mmseg2020,
  title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
  author={MMSegmentation Contributors},
  howpublished = {\url{[https://github.com/open-mmlab/mmsegmentation](https://github.com/open-mmlab/mmsegmentation)}},
  year={2020}
}
```

```bash
@article{mmengine2022,
  title   = {{MMEngine}: OpenMMLab Foundational Library for Training Deep Learning Models},
  author  = {MMEngine Contributors},
  howpublished = {\url{https://github.com/open-mmlab/mmengine}},
  year={2022}
}
```
```bash
@misc{2021mmrazor,
    title={OpenMMLab Model Compression Toolbox and Benchmark},
    author={MMRazor Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmrazor}},
    year={2021}
}
```
# License

This project is released under the [Apache 2.0 license](https://github.com/open-mmlab/mmsegmentation/blob/main/LICENSE "Apache 2.0 license") of mmsegmentation.
