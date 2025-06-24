import os
import subprocess
import json
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import multiprocessing
import random
from vllm import LLM, SamplingParams
from copy import deepcopy
from tqdm import tqdm
import openai
import google.generativeai as genai
import anthropic

def get_parser():
    parser = argparse.ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, type = str, help = 'name of the task')
    parser.add_argument('--device', default = '-1', type = str, help = 'device id(s)')
    return parser


def construct_prompt(data, task, config, stage):
    task_settings = json.load(open(f'task/{task}.json'))

    if stage == '':
        with open(f"prompt/{task}_{config['mode']}.txt") as f:
            prompt_template = f.read()
    else:
        with open(f"prompt/{task}_{config['mode']}_{stage}.txt") as f:
            prompt_template = f.read()

    format_dict = {'relations': json.dumps(task_settings['relations'])}
    
    if task_settings['same_entity_space']:
        format_dict['entities'] = json.dumps(task_settings['entities'])
    else:
        format_dict['head_entities'] = json.dumps(task_settings['head_entities'])
        format_dict['tail_entities'] = json.dumps(task_settings['tail_entities'])
    for a in task_settings['attributes']:
        format_dict['attribute_'+a] = json.dumps(task_settings['attributes'][a])
    if config['mode'] == 'icl':
        annotated_data = json.load(open(f'annotated_data/{task}.json'))
        random.shuffle(annotated_data)
        examples = annotated_data[:config['example_num']]
        example_template = "Text:\n{text}\n\nOutput:\n{interactions}\n\n"
        example_str = "".join([example_template.format(text=i['content'], interactions=i['interactions']) for i in examples]).strip()
        format_dict['examples'] = example_str

    prompts = []
    for i in data:
        cur_format_dict = deepcopy(format_dict)
        cur_format_dict['text'] = i['content']
        prompts.append(prompt_template.format(**cur_format_dict))

    return prompts


def construct_training_data(task, config, stage=''):
    annotated_data = json.load(open(f'annotated_data/{task}.json'))
    prompts = construct_prompt(annotated_data, task, config, stage)

    training_data = []
    for i, prompt in zip(annotated_data, prompts):
        if stage == 'stage2' and len(i['interactions']) == 0:
            continue

        if stage == 'stage1':
            if len(i['interactions']) > 0:
                cur_output = 'Yes'
            else:
                cur_output = 'No'
        else:
            cur_output = json.dumps(i['interactions'])

        cur = i.copy()
        cur['instruction'] = prompt
        cur['input'] = ''
        cur['output'] = cur_output
        training_data.append(cur)

    if not os.path.exists('output/training_data/'):
        os.makedirs('output/training_data/')
    output_name = f'{task}.json' if stage == '' else f'{task}_{stage}.json'
    with open(f'output/training_data/{output_name}', 'w') as f:
        f.write(json.dumps(training_data, indent=2))


def construct_inference_prompts(task, config, stage=''):
    inference_data = json.load(open(f'inference_data/{task}.json'))
    prompts = construct_prompt(inference_data, task, config, stage)

    if stage == 'stage2':
        stage1_pred = json.load(open(f"output/prediction/{task}_{config['mode']}_stage1.json"))
    
    inference_prompts = []
    for idx, i in enumerate(inference_data):
        i['prompt'] = prompts[idx]
        if stage == 'stage2':
            if 'yes' in stage1_pred[idx]['prediction'].lower():
                inference_prompts.append(i)
        else:
            inference_prompts.append(i)

    if not os.path.exists('output/inference_prompts/'):
        os.makedirs('output/inference_prompts/')
    output_name = f'{task}.json' if stage == '' else f'{task}_{stage}.json'
    with open(f'output/inference_prompts/{output_name}', 'w') as f:
        f.write(json.dumps(inference_prompts, indent=2))


def construct_training_config(task, config, stage):
    if config['full_parameter'] == True:
        with open('src/xtuner_config_templates/full_parameter.py') as f:
            xtuner_template = f.read()
    else:
        with open('src/xtuner_config_templates/lora.py') as f:
            xtuner_template = f.read()

    training_data_path = f'output/training_data/{task}.json' if stage == '' else f'output/training_data/{task}_{stage}.json'
    xtuner_config = xtuner_template.replace('{model_path}', '"'+config['model']+'"').replace('{data_path}', '"'+training_data_path+'"').replace('{max_length}', str(config['max_length']))

    if not os.path.exists('output/training_configs/'):
        os.makedirs('output/training_configs/')
    output_name = f'{task}.py' if stage == '' else f'{task}_{stage}.py'
    with open(f'output/training_configs/{output_name}', 'w') as f:
        f.write(xtuner_config)

    return output_name


def convert_model(model_name, training_config_path, task, stage):
    ## path to the last checkpoint
    log_path = f'work_dirs/{task}/last_checkpoint' if stage == '' else f'work_dirs/{task}_{stage}/last_checkpoint'
    ## read the first line of the log file
    with open(log_path) as f:
        checkpoint_file = f.readline()
    model_output_path = f'checkpoint/{model_name}_{task}_{stage}'

    command = ['xtuner', 'convert', 'pth_to_hf', training_config_path, checkpoint_file, model_output_path]

    try:
        result = subprocess.run(
            command, 
            check=True, 
            stdout=sys.stdout, 
            stderr=subprocess.STDOUT
        )
        print(f"Training Succeed")

    except subprocess.CalledProcessError as e:
        print(f"{e.stderr}")


def generate_api(data_slice, cpu_id, config, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, f'response_{cpu_id}.json')
        
    if 'gemini' in config['model']:
        config = genai.GenerationConfig(candidate_count=1,
                      max_output_tokens = config['max_output_length'],
                      temperature = 0.0)
        gemini_model = genai.GenerativeModel(config['model'], generation_config = config)
    elif 'claude' in config['model']:
        client = anthropic.Anthropic()
    elif 'deepseek' in config['model']:
        client = openai.OpenAI(api_key=os.getenv('DEEPSEEK_API_KEY') , base_url="https://api.deepseek.com")

    ret = []
    for i in tqdm(range(len(data_slice)) , desc=f'{cpu_id}'):
        input_text = data_slice[i]['prompt']
        temp = 3
        while temp > 0:
            try:
                if 'gpt' in config['model']:
                    completion = openai.chat.completions.create(
                        model = config['model'],
                        messages = [{'role' : 'user', 'content' : input_text}],
                        temperature = 0.0,
                        max_tokens = config['max_output_length']
                    )
                    answer = completion.choices[0].message.content
                
                elif 'gemini' in config['model']:
                    completion = gemini_model.generate_content(input_text)
                    answer = completion.text.strip()
                elif 'claude' in config['model']:
                    message = client.messages.create(
                        model = config['model'],
                        max_tokens = config['max_output_length'],
                        temperature = 0.0,
                        messages = [{'role' : 'user', 'content' : input_text}]
                    )
                    answer = message.content[0].text
                elif 'deepseek' in config['model']:
                    completion = client.chat.completions.create(
                        model = config['model'],
                        max_tokens = config['max_output_length'],
                        temperature = 0.0,
                        messages = [{'role' : 'user', 'content' : input_text}],
                        stream=False
                    )
                ret.append(answer)
                break
            except Exception as e:
                temp -= 1
                if temp == 0:
                    print(e)
                
        if temp > 0:
            continue
        else:
            ret.append('None')

    json.dump(ret, open(output_file, 'w'), indent=2)



def run_inference_api(data, task, config, stage):
    output_folder = 'output/prediction/tmp/'
    
    pool = multiprocessing.Pool(16)
    results_unmerged = []
    itv = 20

    for sid in list(range(0, len(data), itv)):
        results_unmerged.append(
            pool.apply_async(generate_api, (data[sid:sid+itv], sid, config, output_folder))
        )

    pool.close()
    pool.join()

    # merge all the responses
    all_files = os.listdir(output_folder)
    all_files = [os.path.join(output_folder, file) for file in all_files if '.json' in file]
    all_files.sort(key=lambda i: int(i.split('response_')[1].split('.json')[0]))
    all_responses = []
    for file in all_files:
        all_responses.extend(json.load(open(file)))

    result = data.copy()
    for i in range(len(all_responses)):
        result[i]['prediction'] = all_responses[i]

    output_name = f"{task}_{config['mode']}.json" if stage == '' else f"{task}_{config['mode']}_{stage}.json"
    with open('output/prediction/'+output_name, 'w') as f:
        json.dump(result, f, indent=4)
    
    for file in all_files:
        os.remove(file)


def run_inference_one_gpu(gpu_id, data, model_name, gpu_utilization = 0.9):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(model_name, gpu_memory_utilization= gpu_utilization)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0, max_tokens=4096, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
    
    input_prompts = [d['prompt'] for d in data]
    
    outputs = llm.generate(input_prompts, sampling_params)

    for idx, output in enumerate(outputs):
        data[idx]['prediction'] = output.outputs[0].text.strip()
                        
    return data


def run_inference_multi_gpu(data, args, config, stage, gpu_utilization = 0.9):
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    for i in data:
        i['prompt'] = tokenizer.apply_chat_template([{'role': 'user', 'content': i['prompt']}], tokenize = False, add_generation_prompt = True)
    
    gpu_ids = args.device.split(',')
    num_gpus = len(gpu_ids)
    os.environ['WORLD_SIZE'] = str(num_gpus)
    print(f'Using {num_gpus} GPUs')

    split_list = lambda l, n: [l[i * len(l) // n: (i + 1) * len(l) // n] for i in range(n)]
    split_prompts = split_list(data, num_gpus)
    data_splits = [(gpu_ids[i], p, config['model'], gpu_utilization) for i, p in enumerate(split_prompts)]

    with multiprocessing.Pool(processes=num_gpus) as pool:
        results = pool.starmap(run_inference_one_gpu, data_splits)

    outputs = []
    for result in results:
        outputs.extend(result)

    output_name = f"{args.task}_{config['mode']}.json" if stage == '' else f"{args.task}_{config['mode']}_{stage}.json"
    with open('output/prediction/'+output_name, 'w') as f:
        json.dump(outputs, f, indent=4)


def merge_output(task, config):
    stage1_pred = json.load(open(f"output/prediction/{task}_{config['mode']}_stage1.json"))
    stage2_pred = json.load(open(f"output/prediction/{task}_{config['mode']}_stage2.json"))

    idx_stage2 = 0
    outputs = []
    for i in stage1_pred:
        cur = deepcopy(i)
        cur.pop('prompt', None)
        if 'yes' in cur['prediction'].lower():
            cur['prediction'] = stage2_pred[idx_stage2]['prediction']
            idx_stage2 += 1
        else:
            cur['prediction'] = '[]'
        outputs.append(cur)
    assert idx_stage2 == len(stage2_pred)

    with open(f"output/prediction/{task}_{config['mode']}.json", 'w') as f:
        json.dump(outputs, f, indent=4)
