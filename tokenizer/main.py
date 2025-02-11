from BPETokenizer import BPETokenizer
from dataset import process_dataset
from time import time
from hyperparams import TARGET_VOCABULARY_SIZE

sample = "Wikimedia Commons is a media file repository making available public domain and freely licensed educational media content (images, sound and video clips) to everyone, in their own language. It acts as a common repository for the various projects of the Wikimedia Foundation, but you do not need to belong to one of those projects to use media hosted here. The repository is created and maintained not by paid archivists, but by volunteers. The scope of Commons is set out on the project scope pages. Wikimedia Commons uses the same wiki-technology as Wikipedia and everyone can edit it. Unlike media files uploaded to other projects, files uploaded to Wikimedia Commons can be embedded on pages of all Wikimedia projects without the need to separately upload them there."


def __main__():
    generalTokenizer = BPETokenizer(275, 1024)
    start = time()
    # preprocessed_data = process_dataset()
    generalTokenizer.train(sample)

    print(f"Training took: {time()-start:.2f} seconds")
    print(generalTokenizer.encode(sample))


__main__()
