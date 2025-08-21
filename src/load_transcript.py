import os
import sys
from typing import List, Dict
import re
import string
from multiprocessing import Pool

from nltk.corpus import words
import nltk
import numpy as np
import pymupdf
import pymupdf4llm


INPUT_DIR = "data/pdf"


nltk.download("words", quiet=True)
english_vocab = set(words.words())


### Helpers ###


def is_distribution_skewed(
    occurences: Dict[str, int],
    entropy_threshold: float = 0.7,
    dominant_threshold: float = 0.5,
) -> bool:
    """
    Parameters:
        occurences (dict): Number of occurences of each token.
        entropy_threshold (float): Maximum entropy for a distribution to be considered skewed.
        dominant_threshold (float): Minimum most frequent token ratio for a distribution to be
            considered skewed.
    """
    counts = np.array(list(occurences.values()))
    total = counts.sum()

    probabilities = counts / total
    non_zero_probs = probabilities[probabilities > 0]
    entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
    max_entropy = np.log2(len(occurences))

    normalized_entropy = entropy / max_entropy
    if normalized_entropy < entropy_threshold:
        return True

    dominant_ratio = max(probabilities)
    if dominant_ratio > dominant_threshold:
        return True

    return False


# Try to guess if the string is gibberish using simple heuristics (no ML)
def is_gibberish(page_text: str, threshold: float = 0.5) -> bool:
    tokens = page_text.split()
    if not tokens:
        return True  # Remove empty pages

    alpha_tokens = [token.lower() for token in tokens if token.isalpha()]
    if len(alpha_tokens) == 0:
        return True  # Remove pages with no letters

    # Some pages are just dictionaries of words in alphabetical order; these are considered
    # gibberish. In order to find these, we look at the first letter of each word and ensure
    # their occurances follow a power law distribution; if not, they are likely skewed by
    # one of these pages.
    first_letter_counts = {}
    for token in alpha_tokens:
        if len(token) == 1 and token not in ["a", "i"]:  # Ignore single-letter tokens
            continue
        c = token[0]
        if c not in first_letter_counts:
            first_letter_counts[c] = 0
        first_letter_counts[c] += 1
    if len(first_letter_counts) < 4:
        return True  # If there's only a few first letters, consider it skewed
    if is_distribution_skewed(first_letter_counts):
        return True

    valid_count = 0
    total_tokens = 0

    for token in tokens:
        cleaned = token.strip(string.punctuation)
        if not cleaned:
            continue

        total_tokens += 1

        # english words
        if cleaned.lower() in english_vocab:
            valid_count += 1

        # name, place, etc
        elif cleaned[0].isupper() and cleaned[1:].islower():
            valid_count += 1

        # acronyms
        elif cleaned.isupper() and len(cleaned) <= 4:
            valid_count += 1

        # numbers (i.e., line nubmers)
        elif cleaned.isdigit():
            valid_count += 1

        # sequences of dots (· · · · · ·) show up often and are valid
        elif token.strip() == "·":
            valid_count += 1

        # underlines (i.e., for a signature)
        elif all(c == "_" for c in token):
            valid_count += 1

    valid_ratio = valid_count / total_tokens if total_tokens > 0 else 0
    return valid_ratio < threshold


# Checks for partially or fully alphanumeric strings
def is_alphanumeric(s: str) -> bool:
    return bool(re.search(r"[a-zA-Z0-9]", s))


# If the string has digits and symbols and no alphabetic characters, it's likely a timestamp
def is_probably_timestamp(line: str) -> bool:
    line = line.strip()
    has_digits = bool(re.search(r"\d", line))
    has_alphas = bool(re.search(r"[a-zA-Z]", line))
    has_non_alphanumerics = bool(re.search(r"[^a-zA-Z0-9]", line))

    return has_digits and has_non_alphanumerics and not has_alphas


### Processors ###


def make_single_spaced(text: str) -> str:
    """
    Removes blank lines if and only if it is between two non-blank, alphanumeric lines.
    """

    lines = text.split("\n")
    n = len(lines)

    indices_to_remove = set()

    for i in range(1, n - 1):
        if lines[i].strip() == "":
            prev_line = lines[i - 1]
            next_line = lines[i + 1]

            if is_alphanumeric(next_line) and is_alphanumeric(prev_line):
                indices_to_remove.add(i)

    return "\n".join(line for i, line in enumerate(lines) if i not in indices_to_remove)


def remove_timestamps(text: str) -> str:
    """
    Removes timestamps from the text.
    """

    return "\n".join(
        [line for line in text.split("\n") if not is_probably_timestamp(line)]
    )


def remove_gibberish_pages(text: str) -> str:
    """
    Removes gibberish from the text.
    """

    page_indicies_to_remove: List[int] = []
    pages = text.split("-----")
    for i, page in enumerate(pages):
        if is_gibberish(page):
            page_indicies_to_remove.append(i)

    return "-----".join(
        [page for i, page in enumerate(pages) if i not in page_indicies_to_remove]
    )


### Main functionality ###


def load_transcript(persona_name: str, transcript_key: str) -> None:
    file_path = os.path.join(INPUT_DIR, persona_name, f"{transcript_key}.pdf")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    with open(file_path, "rb") as f:
        pdf_stream = f.read()

    with pymupdf.open(stream=pdf_stream, filetype="pdf") as doc:
        transcript = pymupdf4llm.to_markdown(doc)

    transcript = make_single_spaced(transcript)
    transcript = remove_timestamps(transcript)
    transcript = remove_gibberish_pages(transcript)

    output_dir = os.path.join("data", "personas", persona_name, transcript_key)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "transcript.txt"), "w") as f:
        f.write(transcript)


def main(persona_name: str, transcript_keys: List[str]) -> None:
    transcript_pairs = [(persona_name, key) for key in transcript_keys]
    with Pool(processes=min(len(transcript_pairs), os.cpu_count() or 1)) as pool:
        pool.starmap(load_transcript, transcript_pairs)


if __name__ == "__main__":
    persona_name = sys.argv[1]
    transcript_keys = sys.argv[2:]
    main(persona_name, transcript_keys)
