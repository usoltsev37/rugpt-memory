import argparse
import transformers
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback

from src.models.ltm_gpt.ltm_gpt import LTM_GPT
from src.utils.train_config import load_config
from src.models.load_base_model import load_base_model
from src.data.load_codeparrot_dataset import load_codeparrot_dataset
from src.utils.logger_singleton import logger

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
os.environ['CURL_CA_BUNDLE'] = ''

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='configuration file path'
    )
    args = parser.parse_args()

    main_config = load_config(args.config)

    model, tokenizer = load_base_model(main_config)

    cnt_blocks_with_memory = 2

    model = LTM_GPT(
        model,
        cnt_blocks_with_memory=cnt_blocks_with_memory
    )
 
    # Unfreeze last leyers of model weights
    for param in model.transformer.h[-cnt_blocks_with_memory:].parameters():
        param.requires_grad=True
        # param.data = param.data.to(torch.float32)

    for param in model.transformer.ln_f.parameters():
        param.requires_grad=True
        # param.data = param.data.to(torch.float32)

    for param in model.lm_head.parameters():
        param.requires_grad=True
        # param.data = param.data.to(torch.float32)

    prompts = ['import', 'from', 'while', 'try', 'if', 'for',
               'torch']  # feel free to add a few more that are not 100% assiciated with Python

    MAX_STEPS = 100

    def custom_generate(prompt, model, device, max_steps):
        batch = tokenizer(prompt, return_tensors='pt', return_token_type_ids=False).to(device)
        print(batch)

        for i in range(max_steps):
            outputs = model(**batch)
            # print(outputs)
            probs = outputs.logits[0, -1].nan_to_num(nan=0.0).div(0.8).softmax(-1)  # .argmax(-1).reshape(1, 1)
            old_token = outputs.logits[0, -1].argmax(-1).reshape(1, 1)
            # print(old_token)
            next_token = torch.multinomial(probs, 1).reshape(1, 1)
            # print(next_token)
            batch['input_ids'] = torch.cat([batch['input_ids'], next_token], dim=-1)
            batch['attention_mask'] = torch.cat([batch['attention_mask'], torch.ones_like(next_token)], dim=-1)
            break

        return tokenizer.decode(batch['input_ids'][0].cpu().numpy().tolist()[1:])

    device = torch.device("cuda")

    after_finetuning_samples = []
    for prompt in prompts:
        after_finetuning_samples.append(custom_generate(prompt, model, device, MAX_STEPS))
    print(after_finetuning_samples)


    # dataset = load_codeparrot_dataset(tokenizer)

    # logger.info('Initializing tensorboard')
    # tensorboard_logs_dir = f'logs/tensorboard/{main_config.exp_name}/'
    # tensorboard_writer = SummaryWriter(tensorboard_logs_dir)
    #
    # logger.info('Training')
    # model._hf_peft_config_loaded = True # silence warnings - for research it is ok
    # model.config.use_cache = False # silence warnings from torch
    #
    # trainer_args = main_config.trainer_args
    # trainer = transformers.Trainer(
    #     model=model, train_dataset=dataset['train'],
    #     args=transformers.TrainingArguments(
    #         per_device_train_batch_size=trainer_args.per_device_train_batch_size,
    #         gradient_accumulation_steps=trainer_args.gradient_accumulation_steps,
    #         warmup_steps=trainer_args.warmup_steps,
    #         max_steps=trainer_args.max_steps,
    #         learning_rate=trainer_args.learning_rate,
    #         fp16=trainer_args.fp16,
    #         logging_steps=trainer_args.logging_steps,
    #         output_dir=f'checkpoints/{main_config.exp_name}/outputs',
    #         report_to=trainer_args.report_to,
    #         gradient_checkpointing=True,
    #         gradient_checkpointing_kwargs={'use_reentrant':True}
    #     ),
    #     data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    #     callbacks=[TensorBoardCallback(tb_writer=tensorboard_writer)]
    # )
    # trainer.train()
    #
    # model.config.use_cache = True
    


if __name__ == '__main__':
    main()
