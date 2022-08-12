# Transformer Encoder for Social Science (TESS)

TESS is a deep neural network model intended for social science related NLP tasks. The model is developed by Haosen Ge, In Young Park, Xuancheng Qian, and Grace Zeng. 

The pretrained model weights can be found on Hugging Face: [TESS_768_v1](https://huggingface.co/hsge/TESS_768_v1).

Working paper coming soon ...

<h2>Training Corpus</h2>

|     TEXT      |    SOURCE     |
| ------------- | ------------- |
| Preferential Trade Agreements  | ToTA  |
| Congressional Bills  | BillSum  |
|UNGA Resolutions | UN |
|Firms' Annual Reports | Loughran and McDonald (2016)|
| U.S. Court Opinions | Caselaw Access Project|

The model is trained on 4 NVIDIA A100 GPUs for 120K steps.
