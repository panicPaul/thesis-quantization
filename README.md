# Thesis Project

Flame-driven DF
flame[t] -> encoder -> quantizer -> decoder -> flame[t] (-> flame df)

Audio-driven DF
Audio[t] -> encoder -> quantizer -> decoder -> flame[t] (-> flame df)


per_gaussian_feature -> flame_vertices

s03, t94 has eyes half closed
# To Do's:

- [ ] make single view dataset an infinite dataset; should help with speed issues
- [ ] depth view toggle in viewer
- [ ] sh coloring
- [ ] debugging of whatever causes the training to be so slow

- [ ] use image masks to create torso/ neck code
- [ ] use color masks for lip loss

management



## Installation

To install the repository, run the following commands:

```bash
conda env create --file environment.yml
conda activate thesis_quantization
pre-commit install
```
