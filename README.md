# code for the paper "ESIE-BERT: Enriching Sub-words Information Explicitly with BERT for Intent Classification and Slot Filling"

## Quick Start

Follow the steps below to set up and run our model quickly. 

1. Download the pre-training model from Transformers.

2. Once the model/s you want have been downloaded, you can run the model locally using the command below:
```bash
python train.py \
    --batch_size = 128 \
    --lr = 5e-5 \
    --epoch = 10 \
    --dataset_type_id = 0 
```
**Note**
- Reasonably adjust the following parameters based on your own computational resources to control the model's computational load and parameter count.

huggingface docx: \
https://huggingface.co/docs/transformers/index

pre-train model download URL as following:

### BERT

https://www.huggingface.co/bert-base-uncased \
https://huggingface.co/bert-large-uncased \
https://www.huggingface.co/bert-base-multilingual-uncased

### gpt
https://www.huggingface.co/gpt2  \
https://huggingface.co/gpt2-large   

### t5
https://www.huggingface.co/t5-base \
https://www.huggingface.co/t5-large


mutil-language datasets: \
https://github.com/Makbari1997/Persian-Atis
