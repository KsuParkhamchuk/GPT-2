import pyarrow as pa
import pyarrow.parquet as pq
import csv
import re
import torch
from torch.utils.data import Dataset
from hyperparams import CONTEXT_SIZE

wikitext_file = "datasets/wikitext/train1.arrow"
recipeNLG_file = "datasets/recipe_ngl.csv"
foodfacts_file = "datasets/foodfacts.tsv"
pubmed_sport_file = "datasets/pubmed_sport.parquet"


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


def process_list(list_string: str) -> str:
    items = re.findall(r'"([^"]+)"', list_string)
    joined = ", ".join(item.strip() for item in items)
    return clean_text(joined)


def clean_recipengl(text: str) -> str:
    """
    Processes the inner list string (without the outer brackets) by extracting items within quotes.
    Then, joins the items with a comma and a space and cleans the result.
    """
    # Regex pattern:
    #   1. Capture the title (everything before the first '[')
    #   2. Capture the three bracketed components respectively.

    pattern = r"^(.*?)\[(.*?)\]\[(.*?)\]\[(.*?)\]\s*$"
    match = re.match(pattern, text)
    if not match:
        return text  # or raise an error if the format is unexpected

    title, ingredients_str, instructions_str, labels_str = match.groups()

    # Clean each component:
    title = clean_text(title)
    ingredients_sentence = process_list(ingredients_str)
    instructions_sentence = process_list(instructions_str)
    labels_sentence = process_list(labels_str)

    # Combine each column into its own line.
    cleaned_text = (
        f"{title}\n"
        f"{ingredients_sentence}\n"
        f"{instructions_sentence}\n"
        f"{labels_sentence}"
    )
    return cleaned_text


def clean_pubmed(text) -> str:
    # Remove lines that start with a number followed by a dot (e.g. "96. ...")
    text = re.sub(r"^\d+\.\s.*$", "", text, flags=re.MULTILINE)

    # Remove lines that begin with DOI, PMCID, or PMID (ignoring case)
    text = re.sub(r"(?i)^(DOI|PMCID|PMID):.*$", "", text, flags=re.MULTILINE)

    # Remove the "Author information:" line (ignoring case)
    text = re.sub(
        r"(?si)^Author information:.*?(?=\n\s*\n|$)", "", text, flags=re.MULTILINE
    )

    # Remove copyright notices, e.g. "© 2021 The Society for the Study of Evolution."
    text = re.sub(r"©\s*\d{4}\s*[^\n]*", "", text, flags=re.MULTILINE)

    text = text.strip().replace("\n", " ")
    # Remove extra blank lines created after cleaning
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_text = "\n".join(lines)

    return cleaned_text


def process_arrow(devider):
    # Read the Arrow file using a memory map for efficiency
    with pa.memory_map(wikitext_file) as source:
        arrow_table = pa.ipc.open_stream(source).read_all()

    n_rows_cut = arrow_table.num_rows // devider
    cut_arrow_table = arrow_table.slice(0, n_rows_cut)

    content = cut_arrow_table.column("text")
    content_to_str = "".join(content.to_pylist())
    cleaned_data = clean_wikitext(content_to_str).encode("utf-8")

    return cleaned_data


def process_csv():
    filters = ["title", "ingredients", "directions", "NER"]
    filtered_data = ""
    with open(recipeNLG_file, "r") as source:
        csv_reader = csv.DictReader(source)

        for row in csv_reader:
            clean_row = clean_recipengl("".join(row[col] for col in filters))
            filtered_data.join(clean_row)

    return filtered_data.encode("utf-8")


def process_scraped_text():

    with open(pubmed_sport_file, "rb") as source:
        table = pq.read_table(source)

    content = "".join(item.as_py() for item in table.column("articles"))

    print(content.encode("utf-8"))

    return content.encode("utf-8")


class TextDataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.context_size = CONTEXT_SIZE
        text = process_arrow(1)
        self.tokens = self.tokenizer.encode(text)
        self.total_chunks = len(self.tokens) - CONTEXT_SIZE
        print(self.tokens)

    # required by torch Dataset to stop iteration
    def __len__(self):
        return self.total_chunks

    # required by torch Dataset to get a specific sample of size CONTEXT_SIZE
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.context_size]
        return torch.tensor(chunk)
