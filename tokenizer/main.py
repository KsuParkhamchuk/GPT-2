from BPETokenizer import BPETokenizer
from dataset import process_arrow, process_csv, process_scraped_text
from time import time
from hyperparams import TARGET_VOCABULARY_SIZE


def train_general():
    general_tokenizer = BPETokenizer()
    preprocessed_data = process_arrow()
    general_tokenizer.train(preprocessed_data)


def train_domain_specific():
    domain_specific_tokenizer = BPETokenizer()
    recipengl_data = process_csv()
    pubmed_data = process_scraped_text()
    combined_data = recipengl_data.join(pubmed_data)
    domain_specific_tokenizer.train(combined_data)


def __main__():
    start = time()
    train_domain_specific()
    print(f"Training took: {time()-start:.2f} seconds")


__main__()
