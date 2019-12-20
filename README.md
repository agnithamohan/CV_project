## Enhancing future instance segmentation predictions using Probabilistic U-Nets


### Overview
Our paper adopts a base F2F model that uses Feature Pyramid Networks to predict semantic future maps and attempts to enhance this network by using a Probabilistic U-net to resolve the inherent ambiguities in the input feature maps.

### Usage
evaluate.py [-h] [-L LAYER] [-M MODEL] [-I INPUT] [-O OUTPUT]
```
optional arguments:
  -h, --help            show this help message and exit
  -L LAYER, --layer LAYER
                        : the fpn layer to train
  -M MODEL, --model MODEL
                        : the model checkpoint to load
  -I INPUT, --input INPUT
                        : the input file to evaluate
  -O OUTPUT, --output OUTPUT
                        : the output file
```
*Before Evaluation:*
- Download the pretrained models or use train_model.py to train the models at each layer.

*To Evaluate the model:*
- Run the evaluate.py file using the following command

```python evaluate.py --layer fpn_res3_3_sum --model /models/model.pth --input ./examples/frankfurt_000001_002634.pt --output ./out.pt```
