import os
import subprocess
import json
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import multiprocessing

from utils import *


def train_model(args, config, model_name, stage=''):
    construct_training_data(args.task, config, stage)
    training_config_name = construct_training_config(args.task, config, stage)
    training_config_path = f'output/training_configs/{training_config_name}'
    print(f'Running {training_config_path} on device {args.device}')
    
    command = ['xtuner', 'train', training_config_path, '--deepspeed', 'deepspeed_zero2']

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.device
    env['NPROC_PER_NODE'] = str(len(args.device.split(',')))

    try:
        result = subprocess.run(
            command, 
            check=True, 
            stdout=sys.stdout, 
            stderr=subprocess.STDOUT,
            env=env
        )

    except subprocess.CalledProcessError as e:
        print(f"{e.stderr}")

    convert_model(model_name, training_config_path, args.task, stage)



def run_inference(args, config, stage=''):
    inference_prompts_path = f'output/inference_prompts/{args.task}.json' if stage == '' else f'output/inference_prompts/{args.task}_{stage}.json'
    inference_prompts = json.load(open(inference_prompts_path))

    print(f'Running inference {inference_prompts_path}')
    if config['is_api']:
        run_inference_api(inference_prompts, args.task, config, stage)
    else:
        run_inference_multi_gpu(inference_prompts, args, config, stage)


if __name__ == '__main__':
    args = get_parser().parse_args()
    config = json.load(open(f'config/{args.task}.json'))

    if config['model'].endswith('/'):
        model_name = config['model'].split('/')[-2:-1]
    else:
        model_name = config['model'].split('/')[-1]

    if config['mode'] == 'tune':
        if config['task_decomposition'] == True:
            # training
            # stage 1 model training
            if not os.path.exists(f'checkpoint/{model_name}_{args.task}_stage1/'):
                train_model(args, config, model_name, 'stage1')
            # stage 2 model training
            if not os.path.exists(f'checkpoint/{model_name}_{args.task}_stage2/'):
                train_model(args, config, model_name, 'stage2')

            # inference
            construct_inference_prompts(args.task, config, 'stage1')
            run_inference(args, config, 'stage1')
            construct_inference_prompts(args.task, config, 'stage2')
            run_inference(args, config, 'stage2')
            # merge output of two stages
            merge_output(args.task, config)

        else:
            # training
            if not os.path.exists(f'checkpoint/{model_name}_{args.task}/'):
                train_model(args, config, model_name)

            # inference
            construct_inference_prompts(args.task, config)
            run_inference(args, config)
    else:
        # inference
        construct_inference_prompts(args.task, config)
        run_inference(args, config)

