# Transformer Encoder for Social Science (TESS)

TESS is a deep neural network model intended for social science related NLP tasks. The model is developed by Haosen Ge, In Young Park, Xuancheng Qian, and Grace Zeng. Details of the model can be found in this Paper: [paper](https://www.haosenge.net/_files/ugd/557840_3ce498a7fbc74d6581a947b6c72ef463.pdf).

We demonstrate in two validation tests that TESS outperforms BERT and RoBERTa by 16.7\% on average, especially when the number of training samples is limited (<1,000 training instances). The results display the superiority of TESS on social science text processing tasks. 

The pretrained model weights can be found on Hugging Face: [TESS_768_v1](https://huggingface.co/hsge/TESS_768_v1).


<h2>Training Corpus</h2>

|     TEXT      |    SOURCE     |
| ------------- | ------------- |
| Preferential Trade Agreements  | ToTA  |
| Congressional Bills  | Kornilova and Eidelman (2019)  |
|UNGA Resolutions | UN |
|Firms' Annual Reports | Loughran and McDonald (2016)|
| U.S. Court Opinions | Caselaw Access Project|

The model is trained on 4 NVIDIA A100 GPUs for 120K steps.

