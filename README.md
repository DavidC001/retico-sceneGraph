# retico-sceneGraph

git clone the RelTR repository:
```bash
git clone https://github.com/yrcong/RelTR.git
```
and add it to your Environment Variables `RelTR_PATH` pointing to the cloned repository, if not it will default to `./RelTR`.

download the model from https://drive.google.com/uc?id=1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD

# warnings
current implementation with current libraries give this warning:
```
C:\Users\david\anaconda3\envs\retico_mod\lib\site-packages\torchvision\models\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
C:\Users\david\anaconda3\envs\retico_mod\lib\site-packages\torchvision\models\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
```
This is due to the RelTR code using an older version of torchvision. Newer versions might not work with the current implementation.
 
# References
```
@misc{cong2023reltrrelationtransformerscene,
      title={RelTR: Relation Transformer for Scene Graph Generation}, 
      author={Yuren Cong and Michael Ying Yang and Bodo Rosenhahn},
      year={2023},
      eprint={2201.11460},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2201.11460}, 
}
```