# Thesis Project

Flame-driven DF
flame[t] -> encoder -> quantizer -> decoder -> flame[t] (-> flame df)

Audio-driven DF
Audio[t] -> encoder -> quantizer -> decoder -> flame[t] (-> flame df)


per_gaussian_feature -> flame_vertices

color correction is fucking things up for some reason. (maybe inverted?)


s03, t94 has eyes half closed
# To Do's:

- [ ] depth view toggle in viewer !!!

- [ ] use image masks to create torso/ neck code
- [ ] use color masks for lip loss
- [ ] use per gaussian classification


management



## Installation

To install the repository, run the following commands:

```bash
conda env create --file environment.yml
conda activate thesis_quantization
pre-commit install
```
