# code for paper "ESIE-BERT: Enriching Sub-words Information Explicitly with BERT for Intent Classification and Slot Filling"

## Quick Start

You can follow the steps below to quickly get up and running with Llama 2 models. These steps will let you run quick inference locally. For more examples, see the [Llama 2 recipes repository](https://github.com/facebookresearch/llama-recipes). 

1. In a conda env with PyTorch / CUDA available clone and download this repository.

2. In the top level directory run:
    ```bash
    pip install -e .
    ```
3. Visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and register to download the model/s.

4. Once registered, you will get an email with a URL to download the models. You will need this URL when you run the download.sh script.

5. Once you get the email, navigate to your downloaded llama repository and run the download.sh script. 
    - Make sure to grant execution permissions to the download.sh script
    - During this process, you will be prompted to enter the URL from the email. 
    - Do not use the “Copy Link” option but rather make sure to manually copy the link from the email.

6. Once the model/s you want have been downloaded, you can run the model locally using the command below:
```bash
torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 6
```
**Note**
- Replace  `llama-2-7b-chat/` with the path to your checkpoint directory and `tokenizer.model` with the path to your tokenizer model.
- The `–nproc_per_node` should be set to the [MP](#inference) value for the model you are using.
- Adjust the `max_seq_len` and `max_batch_size` parameters as needed.
- This example runs the [example_chat_completion.py](example_chat_completion.py) found in this repository but you can change that to a different .py file.



```python
python train.py
```

Reasonably adjust the following parameters based on your own computational resources to control the model's computational load and parameter count. Here are some references: \
--batch_size = 128 \
--lr = 5e-5 \
--epoch = 10 \
--dataset_type_id = 0 

huggingface docx: \
https://huggingface.co/docs/transformers/index


pretrain model download url as following:

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
