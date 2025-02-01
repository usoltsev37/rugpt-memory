# RL-based Long-Term Memory for Large Language Models

## TL;DR
This project explores methods to extend the context window of large language models (LLMs) using reinforcement learning (RL) algorithms. A pre-trained Russian-language LLM and a custom dataset derived from Wikipedia texts were utilized. Based on selected quality metrics, the model trained with the REINFORCE algorithm demonstrated comparable performance to the baseline model (a model without memory mechanisms). Further research is needed to develop metrics for evaluating the effectiveness of memory usage.

![Main Scheme](https://github.com/usoltsev37/rugpt-memory/blob/jbelova/materials/main_scheme.jpg)

## Project Materials
All project materials are provided in Russian:
- **Research Text:** [thesis_text.pdf](https://github.com/usoltsev37/rugpt-memory/blob/jbelova/materials/thesis_text.pdf)
- **Presentation:** [presentation.pdf](https://github.com/usoltsev37/rugpt-memory/blob/jbelova/materials/presentation.pdf)

## Dataset Link
You can download dataset [here](https://drive.google.com/drive/folders/1oPf2Yn7NzOYIvqYSJGiXunPvdII73xVH?usp=drive_link).

## Code Navigation
### Train models
Code for models is located in [src/models](https://github.com/usoltsev37/rugpt-memory/tree/jbelova/pretrain_agent/src/models).

The main training code for LLM-LTM is located in [train.py](https://github.com/usoltsev37/rugpt-memory/blob/main/train.py).

For LLM and agent pretraining, see the code in the corresponding branches: [pretrain_ltm](https://github.com/usoltsev37/rugpt-memory/tree/jbelova/pretrain_ltm), [pretrain_agent](https://github.com/usoltsev37/rugpt-memory/tree/jbelova/pretrain_agent).


### RL part
The code related to reinforcement learning can be found in [src/models/rl](https://github.com/usoltsev37/rugpt-memory/tree/jbelova/pretrain_agent/src/models/rl).

The environment, agent, reward function, REINFORCE and agent training are defined there.


