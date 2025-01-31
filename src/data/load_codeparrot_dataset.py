from datasets import load_dataset

from src.utils.logger_singleton import logger


def load_codeparrot_dataset(tokenizer):
    logger.info('Loading dataset')   
    
    dataset = load_dataset("codeparrot/codeparrot-clean-valid", )
    dataset = dataset.map(
        lambda samples: tokenizer(samples['content'][:512]), 
        batched=False
    )
    
    return dataset
