# EALA: Expert-Augmented LLM Annotation for Network Data

## Overview
EALA is designed for annotating network data from text using Large Language Models (LLMs). Our paper ([http://arxiv.org/abs/2503.01672](http://arxiv.org/abs/2503.01672)) demonstrates its application through an example of annotating climate negotiation data.

### How is the task framed?
Given a piece of text $X$, the task is to identify and predict a set of interactions $Y=\{y_1, y_2, \ldots, y_n\}$ among entities mentioned in the text. Each interaction $y_i$ is structured as a quadruplet: $\{e^h_i, e^t_i, r_i, a_i\}$.
* $e^h_i \in E^h$: The head entity (e.g., a sender).
* $e^t_i \in E^t$: The tail entity (e.g., a receiver).
* $r_i \in R$: The relation type describing the interaction between the head and tail entities.
* $a_i = \{a^1_i, a^2_i, \ldots, a^m_i\}$: A set of attributes associated with the interaction, where $a^j_i \in A^j$.

Here, $E^h$, $E^t$, and $R$ represent the predefined label spaces for head entities, tail entities, and relation types, respectively. Similarly, $A^j$ denotes the label space for the $j$-th attribute.

> For example, in the annotation of climate negotiation data, both head and tail entities are countries or coalitions. The relations are categorized into five types: "On behalf of", "Support", "Agreement", "Delaying proposal", and "Opposition". There is also one attribute: the topic in the original annotation, but it is omitted for simplicity in the following description.

### Running Modes
EALA offers two primary modes to suit different data and resource availability: **Instruction Tuning** and **In-Context Learning**.
- **Instruction Tuning**: This mode is ideal for **data-rich scenarios** (typically requiring a thousand or more annotated texts) and necessitates computational resources like GPUs for model training. It's generally recommended for achieving optimal performance.
- **In-Context Learning**: This mode is best suited for **data-scarce scenarios** where only a limited number of examples are available. It's also viable for LLM APIs if you lack dedicated computational resources.

Here's a **roadmap** to help you choose the right mode:
- Do you have sufficient annotated data (approximately >= 1,000 texts with interactions)?
    - Yes:
        - Do you have access to computational resources (e.g., GPUs)?
            - Yes: Use Instruction Tuning.
            - No: Use In-Context Learning.
    - No: Use In-Context Learning.

### How to design the prompt?
The prompt is crucial for guiding the LLM. A well-designed prompt should specify the **task description**, **label spaces**, and the desired **output format**.

To further enhance the model's understanding, it's highly recommended to incorporate a **codebook** into your prompt, including:
- **Label Definitions:** Clear and comprehensive explanations for each label, along with guidelines on when a relation or attribute type should or should not be encoded.
- **Further Coding Rules:** General rules that are not limited to specific label types, such as those related to transitivity and the derivation of existing interactions.

For the **In-Context Learning mode**, please also include **annotated examples** in the prompt to help the model adhere to the instructions and output format.

## Installation
It is recommended to set up a Python-3.10 virtual environment using conda:
```
conda create --name eala python=3.10
conda activate eala
```
Then install packages via pip:
```
pip install -r requirements.txt
```
## Input and Output Formats
### Input
#### Task Settings
Define your task settings in `task/{TASK_NAME}.json`. This file should include the following components:
- `same_entity_space` (boolean): Set to `True` if head and tail entities share the same label space, `False` otherwise.
- `entities` (list of strings): Required if `same_entity_space` is `True`. Lists all entities in the shared entity space.
- `head_entities` (list of strings): Required if `same_entity_space` is `False`. Lists entities in the head entity space.
- `tail_entities` (list of strings): Required if `same_entity_space` is `False`. Lists entities in the tail entity space.
- `relations` (list of strings): Lists all relation types in your relation space.
- `attributes` (list of dictionaries): Each dictionary defines an attribute, where the key is the attribute name and the value is a list of strings representing its label space. Use an empty list `[]` if no attributes are present for the interactions.

> Refer to `task/climate_negotiation.json` for an example.

#### Texts for Inference
Place the texts you wish to annotate in `inference_data/{TASK_NAME}.json`. This file should contain a list of dictionaries, where each dictionary represents a piece of text. The text content is stored under the `content` key, and any other keys can be used for metadata.

> Refer to `inference_data/climate_negotiation.json` for an example.

#### Annotations
The annotated interactions serving for training or in-context learning should be stored in `annotated_data/{TASK_NAME}.json`. This file is a list of dictionaries, with each dictionary representing a text and its corresponding annotations. The `content` key holds the text, the `interactions` key contains a list of dictionaries (each representing a single interaction), and other keys can be used for metadata.

> Refer to `annotated_data/climate_negotiation.json` for an example.

### Output
After running EALA, the predictions will be saved to `output/prediction/{TASK_NAME}_{MODE}.json`. This output file mirrors the structure of your inference data, with an added `prediction` key for each item, containing the predicted interactions.

## Running EALA
### Configuration
Specify your configuration settings in `config/{TASK_NAME}_{MODE}.json`. Key parameters include:
- `model`: The name or path of the LLM to be used. We currently support open-sourced models and APIs of GPT, Gemini, Claude, and Deepseek models. 
- `is_api` (boolean): Whether the model is an API or not.
- `max_output_length`: The maximum length of the generated output.
- `mode`: This should be either `tune` (for instruction tuning) or `icl` (for in-context learning).

For the **instruction tuning mode**, also specify:
- `full_parameter`: A boolean indicating whether to update all model parameters during training. If `false`, LoRA (Low-Rank Adaptation) will be used for more efficient training. You can further customize training details (like changing hyperparameters) in `src/xtuner_config_templates/`.
- `task_decomposition`: A boolean indicating whether to decompose the annotation task into two stages. If `True`, two separate models will be tuned: one to determine if interactions exist in the text, and another to predict the interactions. This is recommended when a large proportion of your texts do not contain any interactions.
- `max_length`: The maximum context length for the model, encompassing both input and output.

For the **instruction tuning mode**, also specify:
- `example_num`: The number of examples presented to the model. The number should be smaller or equal to the number of annotated data, and the examples are randomly sampled from the annotated data.

### Prompt Template
Your prompt template should be located at `prompt/{TASK_NAME}_{MODE}.txt`. A placeholder `{text}` is required for the input text. You can also use placeholders for label spaces: `{entities}` (or `{head_entities}` and `{tail_entities}`), `{relations}`, and `{attribute_ATTRIBUTE_NAME}` (for the label space of a specific attribute). 

If you've selected **task decomposition** for the instruction tuning mode, provide two separate prompts: `prompt/{TASK_NAME}_tune_stage1.txt` and `prompt/{TASK_NAME}_tune_stage2.txt` for the respective stages.

In the **in-context learning mode**, remember to include a `{examples}` placeholder in your prompt for the annotated examples.

> **_Example_**  
> To annotate climate negotiation data in instruction tuning mode using Llama3-8b-instruct with task decomposition, configure `config/climate_negotiation_tune.json` and use prompts in `prompt/climate_negotiation_tune_stage1.txt` and `prompt/climate_negotiation_tune_stage2.txt`.
> To annotate the climate negotiation data in the in-context learning mode using GPT-4o, the config is in `config/climate_negotiation_icl.json`, and the prompt is in `prompt/climate_negotiation_icl.txt`.

### Running the Script
Execute EALA using the following command:
```
python src/run.py --task {TASK_NAME} --device {DEVICE}
```
If you have access to computational resources, specify the GPU device IDs (e.g., 0 or 0,1 for multiple GPUs) in {DEVICE}; othervise set it to -1.

## Citation
If EALA is helpful for your research, please cite our paper:
```
@article{liu2025automated,
  title={Automated Annotation of Evolving Corpora for Augmenting Longitudinal Network Data: A Framework Integrating Large Language Models and Expert Knowledge},
  author={Liu, Xiao and Wu, Zirui and Li, Jiayi and Shao, Zhicheng and Pang, Xun and Feng, Yansong},
  journal={arXiv preprint arXiv:2503.01672},
  year={2025}
}
```
For any questions or issues, feel free to create an issue on this repository or contact us via email.