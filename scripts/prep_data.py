import argparse
import re
import sys
from pathlib import Path



class Token:
    """One word/token from a CoNLL-U file."""
    def __init__(self, token_id, upos, head, deprel, lang):
        self.token_id = token_id  # integer position in sentence
        self.upos = upos          # universal part-of-speech tag (e.g. "NOUN")
        self.head = head          # integer ID of the parent word
        self.deprel = deprel      # dependency relation label (e.g. "nsubj")
        self.lang = lang          # language tag (e.g. "en", "hi")


class Sentence:
    """A full sentence made up of Token objects."""
    def __init__(self, sent_id, tokens):
        self.sent_id = sent_id  # sentence identifier string from the file
        self.tokens = tokens    # list of Token objects


def get_lang(misc):
    """Pull the language code out of the MISC field.
    The MISC column can look like:
      "Lang=te" or "Lang=te|SpaceAfter=No" or "_" or just bare like "en"
    """
    if misc == "_":
        return None
    for part in misc.split("|"):
        if part.startswith("Lang="):
            return part[len("Lang="):]  # everything after "Lang="
    # hi_en stores the language code directly with no key-value structure
    if "=" not in misc:
        return misc
    return None


def get_head(head_str):
    """Convert the HEAD field to an integer"""
    if head_str == "_":
        return None
    try:
        return int(head_str)
    except ValueError:
        return None



def parse_conllu(filepath):
    """Read a CoNLL-U file and return a list of Sentence objects"""
    sentences = []
    current_tokens = []
    current_sent_id = ""

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # Grab the sentence ID from comment lines
            if line.startswith("# sent_id"):
                current_sent_id = line.split("=", 1)[-1].strip()
                continue

            # Skip all other comment lines
            if line.startswith("#"):
                continue

            # A blank line means the sentence is complete
            if line == "":
                if current_tokens:
                    sentences.append(Sentence(current_sent_id, current_tokens))
                    current_tokens = []
                    current_sent_id = ""
                continue

            # Split the token line into its 10 columns
            columns = line.split("\t")
            if len(columns) < 10:
                continue  # skip malformed lines

            token_id_str = columns[0]

            # Skip multi-word tokens like "1-2" and empty nodes like "1.1"
            if "-" in token_id_str or "." in token_id_str:
                continue

            token = Token(
                token_id=int(token_id_str),
                upos=columns[3],
                head=get_head(columns[6]),
                deprel=columns[7],
                lang=get_lang(columns[9]),
            )
            current_tokens.append(token)

    # Handle files that don't end with a blank line
    if current_tokens:
        sentences.append(Sentence(current_sent_id, current_tokens))

    return sentences



# To make it easy to dynamically add more data

# Matches suffixes like "_train", "_test", "_dev", "_val" at the end of a filename stem
split_suffix = re.compile(r"_(train|test|dev|val)$")

def find_datasets(data_dir):
    """Scan data_dir and group .conllu files by language-pair prefix.

    For example:
      te_en_train.conllu  =  "te_en"
      te_en_test.conllu   =  "te_en"   (merged automatically)
      hi_en.conllu        =  "hi_en"

    Returns a dict mapping prefix to list of file paths.
    """
    groups = {}

    for filepath in sorted(data_dir.glob("*.conllu")):
        # Strip split suffix to get the dataset prefix
        prefix = split_suffix.sub("", filepath.stem)

        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(filepath)

    return groups



def remove_punct(sentence):
    """Remove tokens that cannot be code-switch points.

    Excluded:
      - UPOS=PUNCT  — punctuation has no language identity
      - Lang=univ   — explicitly language-neutral tokens
      - Lang=ne     — named entities (e.g. "Delhi", "Obama") in hi_en
      - Lang=acro   — acronyms (e.g. "BJP", "USA") in hi_en
    """
    language_neutral = {"univ", "ne", "acro"}
    return [
        token for token in sentence.tokens
        if token.upos != "PUNCT" and token.lang not in language_neutral
    ]



def check_and_report(prefix, sentences):
    """Print a summary of counts and data issues for one dataset."""

    n_sentences = len(sentences)
    n_tokens_raw = sum(len(s.tokens) for s in sentences)
    n_tokens_kept = sum(len(remove_punct(s)) for s in sentences)

    # Sentences where at least one HEAD is missing
    sentences_missing_head = sum(
        1 for s in sentences
        if any(t.head is None for t in s.tokens)
    )

    # Tokens with no language tag at all
    tokens_missing_lang = sum(
        1 for s in sentences
        for t in s.tokens
        if t.lang is None
    )

    # Monolingual sentences: only one language present (after filtering)
    # Track which language it is
    monolingual_counts = {}
    for sentence in sentences:
        kept = remove_punct(sentence)
        langs = {t.lang for t in kept if t.lang is not None}
        if len(langs) == 1:
            lang = next(iter(langs))
            monolingual_counts[lang] = monolingual_counts.get(lang, 0) + 1

    print(f"\n=== {prefix} ===")
    print(f"  Sentences      : {n_sentences:,}")
    print(f"  Tokens (raw)   : {n_tokens_raw:,}")
    print(f"  Tokens (kept)  : {n_tokens_kept:,}")
    print(f"  Missing HEAD   : {sentences_missing_head:,} sentences")
    print(f"  Missing lang   : {tokens_missing_lang:,} tokens")

    if monolingual_counts:
        print("  Monolingual sentences by language:")
        for lang, count in sorted(monolingual_counts.items()):
            pct = 100.0 * count / n_sentences if n_sentences else 0.0
            print(f"    {lang:<6}: {count:,}  ({pct:.1f} %)")
    else:
        print("  Monolingual sentences: none")



def write_tsv(sentences, output_path):
    """Write the filtered tokens for one dataset to a TSV file."""

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("sent_id\ttoken_id\tupos\thead\tdeprel\tlang\n")

        for sentence in sentences:
            for token in remove_punct(sentence):
                head = str(token.head) if token.head is not None else "_"
                lang = token.lang if token.lang is not None else "_"
                out.write(
                    f"{sentence.sent_id}\t{token.token_id}\t{token.upos}\t"
                    f"{head}\t{token.deprel}\t{lang}\n"
                )




def main():
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data" / "raw",
        help="Folder containing .conllu files (default: data/raw/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "clean",
        help="Folder for output TSV files (default: data/clean/)",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Error: folder not found: {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = find_datasets(args.data_dir)

    if not datasets:
        print(f"No .conllu files found in {args.data_dir}", file=sys.stderr)
        sys.exit(1)

    for prefix, files in sorted(datasets.items()):
        # Parse all files for this dataset and combine into one list
        sentences = []
        for filepath in files:
            sentences.extend(parse_conllu(filepath))

        check_and_report(prefix, sentences)

        output_path = args.output_dir / f"{prefix}_tokens.tsv"
        write_tsv(sentences, output_path)
        print(f" => written to {output_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
