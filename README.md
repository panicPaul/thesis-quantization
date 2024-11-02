# Remaining Tasks
## Audio2Flame
- [ ] try codetalker

## Rendering
- [ ] train on only 8 or so frames and then fine tune the flame vertices on each frame to get a better motion prior
    - copy video class
- [ ] implement per gaussian classification

# Thesis Project

Flame-driven DF
flame[t] -> encoder -> quantizer -> decoder -> flame[t] (-> flame df)

Audio-driven DF
Audio[t] -> encoder -> quantizer -> decoder -> flame[t] (-> flame df)


per_gaussian_feature -> flame_vertices

color correction is fucking things up for some reason. (maybe inverted?)


- [ ] experiment with predicting flame offsets based on audio maybe? Similar to tri plane encoding!!!!



- [ ] local rigidity loss!!!


s03, t94 has eyes half closed
# To Do's:
- [ ] debug the KNN and flame deformation




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
