import argparse
import transformers
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback
import subprocess

from src.utils.train_config import load_config
from src.models.load_base_model import load_base_model
from src.data.load_codeparrot_dataset import load_codeparrot_dataset
from src.utils.logger_singleton import logger

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True, help='configuration file path'
    )
    args = parser.parse_args()

    main_config = load_config(args.config)

    model, tokenizer = load_base_model(main_config)
    
    dataset = load_codeparrot_dataset(tokenizer)
 
    # Unfreeze 6.8% of model weights 
    for param in model.transformer.h[38:].parameters():
        param.requires_grad=True
        param.data = param.data.to(torch.float32)

    for param in model.transformer.ln_f.parameters():
        param.requires_grad=True
        param.data = param.data.to(torch.float32)

    for param in model.lm_head.parameters():
        param.requires_grad=True
        param.data = param.data.to(torch.float32)
    
    logger.info('Initializing tensorboard')
    tensorboard_logs_dir = f'logs/{main_config.exp_name}/tensorboard'
    tensorboard_writer = SummaryWriter(tensorboard_logs_dir)
    subprocess.Popen(
        ['tensorboard', '--logdir', tensorboard_logs_dir]
    )
    
    logger.info('Training')
    trainer_args = main_config.trainer_args
    trainer = transformers.Trainer(
        model=model, train_dataset=dataset['train'],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=trainer_args.per_device_train_batch_size, 
            gradient_accumulation_steps=trainer_args.gradient_accumulation_steps,
            warmup_steps=trainer_args.warmup_steps, 
            max_steps=trainer_args.max_steps, 
            learning_rate=trainer_args.learning_rate, 
            fp16=trainer_args.fp16,
            logging_steps=trainer_args.logging_steps, 
            output_dir=f'checkpoints/{main_config.exp_name}/outputs', 
            report_to=trainer_args.report_to
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[TensorBoardCallback(tb_writer=tensorboard_writer)]
    )
    trainer.train()
    


if __name__ == '__main__':
    main()
