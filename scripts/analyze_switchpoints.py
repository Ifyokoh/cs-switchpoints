import argparse
import sys
from pathlib import Path



class Token:
    """One content token loaded from a cleaned tokens TSV."""
    def __init__(self, token_id, upos, head, deprel, lang):
        self.token_id = token_id  # integer position in sentence
        self.upos = upos          # universal part-of-speech tag (e.g. "NOUN")
        self.head = head          # integer ID of the parent word
        self.deprel = deprel      # dependency relation label (e.g. "nsubj")
        self.lang = lang          # language tag (e.g. "en", "hi")


class Sentence:
    """A sequence of content tokens that belong to the same sentence."""
    def __init__(self, sent_id, tokens):
        self.sent_id = sent_id  # sentence identifier string
        self.tokens = tokens    # list of Token objects, already filtered


class TokenPair:
    """One adjacent content-token pair from a sentence."""
    def __init__(self, sent_id, tok_i, tok_j, switch, relation):
        self.sent_id = sent_id    # sentence the pair came from
        self.tok_i = tok_i        # left token
        self.tok_j = tok_j        # right token (next in sequence)
        self.switch = switch      # True / False / None (None means lang unknown)
        self.relation = relation  # "within", "boundary", or "unknown"



def read_tokens_tsv(filepath):
    """Read a *_tokens.tsv file and return a list of Sentence objects.
    Each row in the TSV is one token. Rows are grouped by sent_id to
    reconstruct sentences. 
    """
    # Collect rows grouped by sent_id, preserving order
    sentence_map = {}
    sentence_order = []

    with open(filepath, encoding="utf-8") as f:
        header = f.readline() 

        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            columns = line.split("\t")
            sent_id = columns[0]
            token_id = int(columns[1])
            upos = columns[2]
            head = int(columns[3])
            deprel = columns[4]
            lang = columns[5] if columns[5] != "_" else None

            token = Token(token_id=token_id, upos=upos, head=head, deprel=deprel, lang=lang)

            if sent_id not in sentence_map:
                sentence_map[sent_id] = []
                sentence_order.append(sent_id)
            sentence_map[sent_id].append(token)

    sentences = [
        Sentence(sent_id, sentence_map[sent_id])
        for sent_id in sentence_order
    ]
    return sentences


def find_datasets(clean_dir):
    """Find all *_tokens.tsv files in clean_dir.
    Returns a dict mapping dataset prefix to file path, e.g.:
      {"hi_en": Path("data/clean/hi_en_tokens.tsv"), ...}
    """
    datasets = {}
    for filepath in sorted(clean_dir.glob("*_tokens.tsv")):
        # Strip "_tokens" suffix to get the dataset prefix
        prefix = filepath.stem[: -len("_tokens")]
        datasets[prefix] = filepath
    return datasets



def is_switch(tok_i, tok_j):
    """Decide whether two adjacent tokens represent a language switch.
    Returns True for a switch, False for same language, None when either
    language tag is missing and we can't tell.
    """
    if tok_i.lang is None or tok_j.lang is None:
        return None
    return tok_i.lang != tok_j.lang


def get_relation(tok_i, tok_j):
    """Decide whether two tokens are inside a constituent or at its boundary.
    A pair counts as "within" when:
      - tok_i directly depends on tok_j  (head of i == id of j), or
      - tok_j directly depends on tok_i  (head of j == id of i), or
      - both tokens share the same head word.
    Returns "within", "boundary", or "unknown" when HEAD data is missing.
    """
    if tok_i.head is None or tok_j.head is None:
        return "unknown"

    # tok_i depends on tok_j
    if tok_i.head == tok_j.token_id:
        return "within"

    # tok_j depends on tok_i
    if tok_j.head == tok_i.token_id:
        return "within"

    # both hang off the same parent word
    if tok_i.head == tok_j.head:
        return "within"

    return "boundary"


def extract_pairs(sentence):
    """Build the list of adjacent token pairs for one sentence."""
    pairs = []

    for k in range(len(sentence.tokens) - 1):
        tok_i = sentence.tokens[k]
        tok_j = sentence.tokens[k + 1]
        pairs.append(
            TokenPair(
                sent_id=sentence.sent_id,
                tok_i=tok_i,
                tok_j=tok_j,
                switch=is_switch(tok_i, tok_j),
                relation=get_relation(tok_i, tok_j),
            )
        )

    return pairs




def check_and_report(prefix, pairs):
    """Print a switch-point summary for one dataset."""

    total = len(pairs)

    # Pairs where we couldn't determine switch status (missing lang tag)
    lang_unknown = sum(1 for p in pairs if p.switch is None)
    known_pairs = [p for p in pairs if p.switch is not None]
    n_known = len(known_pairs)

    same_pairs = [p for p in known_pairs if not p.switch]
    n_same = len(same_pairs)
    n_switch = sum(1 for p in known_pairs if p.switch)

    # Count same-language pairs by language
    same_by_lang = {}
    for p in same_pairs:
        lang = p.tok_i.lang
        same_by_lang[lang] = same_by_lang.get(lang, 0) + 1

    # Break down switch pairs by syntactic relation
    switch_pairs = [p for p in known_pairs if p.switch]
    n_sw_within = sum(1 for p in switch_pairs if p.relation == "within")
    n_sw_boundary = sum(1 for p in switch_pairs if p.relation == "boundary")
    n_sw_unknown_rel = sum(1 for p in switch_pairs if p.relation == "unknown")

    def pct(numerator, denominator):
        if denominator == 0:
            return "  n/a"
        return f"{100.0 * numerator / denominator:.1f} %"

    print(f"\n=== {prefix} ===")
    print(f"  Total adjacent pairs  : {total:>3,}")
    print(f"  Lang-unknown pairs    : {lang_unknown:>3,}  (skipped in switch stats)")
    print(f"  Same-language pairs   : {n_same:>3,}  ({pct(n_same, n_known)})")
    for lang, count in sorted(same_by_lang.items()):
        print(f"    => {lang:<3}              : {count:>3,}  ({pct(count, n_same)} of same-lang)")
    print(f"  Switch pairs          : {n_switch:>3,}  ({pct(n_switch, n_known)})")
    print(f"    => within constituent : {n_sw_within:>3,}  ({pct(n_sw_within, n_switch)} of switches)")
    print(f"    => at boundary        : {n_sw_boundary:>3,}  ({pct(n_sw_boundary, n_switch)} of switches)")
    print(f"    => relation unknown   : {n_sw_unknown_rel:>3,}  ({pct(n_sw_unknown_rel, n_switch)} of switches)")



def write_tsv(pairs, output_path):
    """Write pair-level data to a TSV file."""

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(
            "sent_id\ttok_i_id\ttok_j_id\tupos_i\tupos_j\t"
            "lang_i\tlang_j\tswitch\trelation\n"
        )
        for pair in pairs:
            lang_i = pair.tok_i.lang if pair.tok_i.lang is not None else "_"
            lang_j = pair.tok_j.lang if pair.tok_j.lang is not None else "_"

            if pair.switch is None:
                switch_str = "unknown"
            elif pair.switch:
                switch_str = "true"
            else:
                switch_str = "false"

            out.write(
                f"{pair.sent_id}\t"
                f"{pair.tok_i.token_id}\t{pair.tok_j.token_id}\t"
                f"{pair.tok_i.upos}\t{pair.tok_j.upos}\t"
                f"{lang_i}\t{lang_j}\t"
                f"{switch_str}\t{pair.relation}\n"
            )



def main():
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=repo_root / "data" / "clean",
        help="Folder containing *_tokens.tsv files from prep_data.py (default: data/clean/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "data" / "switchpairs",
        help="Folder for output TSV files (default: data/switchpairs/)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = find_datasets(args.clean_dir)


    for prefix, filepath in sorted(datasets.items()):
        sentences = read_tokens_tsv(filepath)

        # Build all adjacent pairs across every sentence
        pairs = []
        for sentence in sentences:
            pairs.extend(extract_pairs(sentence))

        check_and_report(prefix, pairs)

        output_path = args.output_dir / f"{prefix}_pairs.tsv"
        write_tsv(pairs, output_path)
        print(f"  => written to {output_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
