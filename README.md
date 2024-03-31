# SMILE Dataset (NAACL 2024)
This repository contains dataset and pytorch implementation for the NAACL 2024 paper, [SMILE: A Multimodal Dataset for Understanding Laughter with Language Models](https://arxiv.org/pdf/2312.09818.pdf).

![teaser_v3(1)-1](https://github.com/SMILE-data/SMILE/assets/136543087/13910eab-6603-4c97-ad1f-8098bd1fc67f)

## Installation
```
$ conda create -n SMILE python==3.10.11
$ conda activate SMILE

# move to FastChat/ directory
$ cd FastChat

$ pip3 install --upgrade pip  
$ pip3 install -e .
$ pip3 install openai
$ pip3 install scikit-image
$ pip3 install evaluate
$ pip3 install bert-score
```


### Download the SMILE Dataset
0. Now, we are updating SMILE dataset v.2. After the update, we will update the laugh reasoning benchmark.
1. Download SMILE dataset v.2 in [here](https://drive.google.com/file/d/15KjSeYn3tjiHXiswLgffmxPEoMYtepv2/view?usp=sharing)
2. unzip the dataset.

    ```
    
    ├── annotations
    |    ├── data_split.json
    |    ├── GT_laughter_reason.json
    |    └── multimodal_textual_representation.json
    |
    └── videos
         └── SMILE_videos.zip
                ├── video_clips
                └── video_segments
    
    ``` 
3. Details about each file
   - data_split.json: key index for train, validation, test split
   - GT_laughter_reason.json: Ground-Truth laughter reason for video clip
   - multimodal_textual_representation.json: multimodal textual representation encoded from video clip
   - video_clips: 887 video clips from sitcom and TED, **Note**: sitcom has an underbar in the key index, while TED does not. You can use this information for splitting our dataset by video types.
   - video_segments: 4482 video segments trimmed from video clip by utterances.

4. SMILE dataset v.1 for evaluation
   - We provide v.1 dataset for evaluation download in [hear](https://drive.google.com/file/d/1pPXli0F_2XJWqj1I4SHI93yOu1i1R5jF/view?usp=sharing)
   - Note that sitcom_reasoning_{train/val}.json and ted_reasoning_{train/val}.json are subset of smile_reasoning_{train/val}.json.
     
    ```
    
    ├── SMILE_v1_evaluation
         ├── smile_reasoning_train.json
         ├── smile_reasoning_val.json
         ├── sitcom_reasoning_train.json
         ├── sitcom_reasoning_val.json
         ├── ted_reasoning_train.json
         └── ted_reasoning_val.json
    
    ```

## Evaluation

### Laugh reasoning 
We provide the inference code for in-context and zero-shot experiment using GPT3. 

As the fine-tuneded GPT3 requires a certain openai api-key which the model was fine-tuned on, we instead provide the inferecne code for fine-tuned model using LLaMA. 

Please evaluate the models with the provided v.1. dataset.


#### In-context and Zero-shot experiment (GPT3)
Note that running GPT3 requires your own openai api-key and it also charges for running the model.

Replace the { } with your own information. 

```
$ python gpt3_inferece.py -openai_key {your openai api key} -engine {name of gpt3 model} -shot {fewshot or zeroshot} -val_data {path/for/validation_data} -train_data {path/for/train_data} -random_seed {any integer number} 
```
#### Fine-tuned experiment (LLaMA)
We provide the pre-trained weights of the LLaMA for the research purpose only.

| Training data | Link                                                                                                    |
|--------------|---------------------------------------------------------------------------------------------------------|
| SMILE        | [SMILE_checkpoint](https://www.dropbox.com/sh/0codb006h40mm61/AABHnPAQt2e_JYQNxJiJ9jyFa?dl=0)  |
| SMILE_Sitcom | [Sitcom_checkpoint](https://www.dropbox.com/sh/fbytnml1utm51mb/AAC8a41vKgSCE2LxY-bm77Lsa?dl=0) |
| SMILE_Ted    | [Ted_checkpoint](https://www.dropbox.com/sh/4zzsonu8fo9lbh8/AACJlCxv_hW7DDD0GTadFGE1a?dl=0)    |

   ``` 
    ├── SMILE
         ├── checkpoint
    ├── SMILE_SITCOM
         ├── checkpoint
    ├── SMILE_TED
         ├── checkpoint
   ```


Replace the { } with your own information.

You should direct the checkpoint directory for the model_path, e.g., "SMILE/checkpoint".
```
$ python FastChat/fastchat/serve/inference.py -model_path {path/for/fine-tuned model} -val_data {path/for/validation_data} -train_data {path/for/train_data} -random_seed {any integer number}
```

## Acknowledgement
We are grateful for the following awesome projects, our SMILE arising from:
- [GPT3](https://arxiv.org/pdf/2005.14165.pdf): Language Models are Few-Shot Learners
- [LLaMA](https://github.com/facebookresearch/llama): LLaMA: Open and Efficient Foundation Language Models
- [Vicuna](https://github.com/lm-sys/FastChat): Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality
- [MUStARD](https://github.com/soujanyaporia/MUStARD): Towards Multimodal Sarcasm Detection (An _Obviously_ Perfect Paper)
- [UR-FUNNY](https://aclanthology.org/D19-1211/): UR-FUNNY: A Multimodal Language Dataset for Understanding Humor


