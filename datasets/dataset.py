import pyarrow as pa
import re
import torch
from torch.utils.data import Dataset
from hyperparams import CONTEXT_SIZE


def clean_punctuation_spacing(text: str) -> str:
    """
    Clean spacing around punctuation marks.
    Removes spaces before punctuation and ensures one space after.
    """
    import re

    # List of punctuation marks to handle
    punctuation_marks = [
        ".",
        ",",
        "!",
        "?",
        ":",
        ";",
        ")",
        "]",
        "}",
        '"',
        "'",
    ]

    # Remove spaces before punctuation marks
    for punct in punctuation_marks:
        text = re.sub(rf"\s+\{punct}", punct, text)

    # Handle opening brackets/quotes (remove space after)
    opening_marks = ["(", "[", "{", '"']
    for punct in opening_marks:
        text = re.sub(rf"\{punct}\s+", punct, text)

    # Ensure single space after punctuation (except for periods in numbers)
    text = re.sub(r"([.,!?:;])\s*(?![0-9])", r"\1 ", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_wikitext(text: str) -> str:
    # Remove section markers like "= = Gameplay = =" or "- Valkyria Chronicles III -"
    text = re.sub(r"=\s*=\s*[^=]+\s*=\s*=", "", text)
    text = re.sub(r"(?:(?<=\s)|^)=+\s+([A-Za-z][A-Za-z\s]*?)\s+=+(?=\s|$)", "", text)

    # Remove multiple spaces, newlines, and tabs
    text = re.sub(r"\s+", " ", text)

    # Remove special characters that don't contribute to meaning
    text = re.sub(r"@-@|@@", "", text)

    # Remove references and special markers
    text = re.sub(r"\[\d+\]", "", text)  # Remove [1], [2], etc.

    # Remove spaces arounf punctuation marks
    text = clean_punctuation_spacing(text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def clean_text(s: str) -> str:
    # Remove leading/trailing whitespace
    s = s.strip()
    # Remove any comma that immediately follows a period, with optional spaces in between.
    s = re.sub(r"\.\s*,", ".", s)
    # Normalize multiple spaces to a single space.
    s = re.sub(r"\s+", " ", s)
    return s


def process_arrow(devider, filepath):
    # Read the Arrow file using a memory map for efficiency
    with pa.memory_map(filepath) as source:
        arrow_table = pa.ipc.open_stream(source).read_all()

    n_rows_cut = arrow_table.num_rows // devider
    cut_arrow_table = arrow_table.slice(0, n_rows_cut)

    content = cut_arrow_table.column("text")
    content_to_str = "".join(content.to_pylist())
    cleaned_data = clean_wikitext(content_to_str).encode("utf-8")

    return cleaned_data


class TextDataset(Dataset):
    def __init__(self, tokenizer, filepath):
        self.tokenizer = tokenizer
        self.context_size = CONTEXT_SIZE
        text = process_arrow(1, filepath)
        self.tokens = self.tokenizer.encode(text)
        self.total_chunks = len(self.tokens) - CONTEXT_SIZE

    # required by torch Dataset to stop iteration
    def __len__(self):
        return self.total_chunks

    # required by torch Dataset to get a specific sample of size CONTEXT_SIZE
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.context_size]
        return torch.tensor(chunk)
