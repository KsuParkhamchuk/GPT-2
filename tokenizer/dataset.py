import pyarrow as pa
import pandas as pd
import re


file_path_1 = "tokenizer/wikitext/train1.arrow"
file_path_2 = "tokenizer/wikitext/train2.arrow"


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
        text = re.sub(f"\s+\\{punct}", punct, text)

    # Handle opening brackets/quotes (remove space after)
    opening_marks = ["(", "[", "{", '"']
    for punct in opening_marks:
        text = re.sub(f"\\{punct}\s+", punct, text)

    # Ensure single space after punctuation (except for periods in numbers)
    text = re.sub(r"([.,!?:;])\s*(?![0-9])", r"\1 ", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_data(text):
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


def process_dataset():
    # Read the Arrow file using a memory map for efficiency
    with pa.memory_map(file_path_1) as source:
        arrow_table_1 = pa.ipc.open_stream(source).read_all()

    # with pa.memory_map(file_path_2) as source:
    #     arrow_table_2 = pa.ipc.open_stream(source).read_all()

    # combined_table = pa.concat_tables([arrow_table_1, arrow_table_2])
    df = arrow_table_1.to_pandas()
    cleaned_texts = df["text"].map(clean_data)

    doc_separator = "\n\n<DOC_SEP>\n\n"
    return doc_separator.join(cleaned_texts)
