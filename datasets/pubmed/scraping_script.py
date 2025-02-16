import requests
import pyarrow as pa
import pyarrow.parquet as pq
from dataset import clean_pubmed


def get_pubmed_ids():
    terms = ["workout", "sport", "resistance training", "fitness"]
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    base_params = {
        "db": "pubmed",
        "retmax": 25,
        "retmode": "json",
        "sort": "relevance",
    }

    ids = []

    for term in terms:
        params = base_params.copy()
        params["term"] = term
        response = requests.get(search_url, params=params)
        data = response.json()
        ids.extend(data["esearchresult"]["idlist"])

    return ids


def get_records():
    ids = get_pubmed_ids()
    fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
    params = {"db": "pubmed", "id": ids, "rettype": "abstract", "retmode": "text"}

    response = requests.get(fetch_url, params=params)
    cleaned_data = clean_pubmed(response.text)

    table = pa.table({"articles": [cleaned_data]})
    pq.write_table(table, "tokenizer/datasets/pubmed_sport.parquet")


get_records()
