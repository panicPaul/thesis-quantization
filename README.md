# Thesis Project

Flame-driven DF
flame[t] -> encoder -> quantizer -> decoder -> flame[t] (-> flame df)

Audio-driven DF
Audio[t] -> encoder -> quantizer -> decoder -> flame[t] (-> flame df)


per_gaussian_feature -> flame_vertices

color correction is fucking things up for some reason. (maybe inverted?)


s03, t94 has eyes half closed
# To Do's:
- [ ] debug the KNN and flame deformation


- [ ] dataset that supports flame, audio windowing but not image
- [ ] video training
- [ ] new deformation field



- [ ] video sequence training
- [ ] use per gaussian classification (do I actually need that tho?)
- [ ] nerfview toggle to enable/ disable the classes

- [ ] k-nn search in flame vertices
- [ ] per flame latents
- [ ] attention-aggregation
- [ ] windowed-input (only one time step for the images should be loaded)
- [ ] change post and pre-processing to be their own functions
- [ ] quantize windows (i.e. 512)
- [ ] audio feature with severe bottleneck / small discrete quantization to infer emotions etc.

- [ ] we can use boring windowed predictions to feed it into our model to see how far this gets us

management



## Installation

To install the repository, run the following commands:

```bash
conda env create --file environment.yml
conda activate thesis_quantization
pre-commit install
```
