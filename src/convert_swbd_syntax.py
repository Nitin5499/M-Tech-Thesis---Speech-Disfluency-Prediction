"""
convert_swbd_syntax.py
Converts Switchboard NXT syntax XML files to linetrees format.
Combines with existing NLTK PTB linetrees to create a larger training set.

INPUT:
  data/switchboard-nxt-release-1.0/xml/syntax/   (XML parse trees)
  data/switchboard-nxt-release-1.0/xml/terminals/ (word + POS tags)

OUTPUT:
  data/swbd.linetrees          (Switchboard trees in linetrees format)
  data/combined.linetrees      (NLTK PTB + Switchboard combined)
"""

import os
import re
from lxml import etree

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(BASE_DIR)

SWBD_ROOT  = os.path.join(ROOT_DIR, "data",
             "switchboard-nxt-release-1.0", "xml")
SYNTAX_DIR = os.path.join(SWBD_ROOT, "syntax")
TERM_DIR   = os.path.join(SWBD_ROOT, "terminals")

SWBD_LINES = os.path.join(ROOT_DIR, "data", "swbd.linetrees")
NLTK_LINES = os.path.join(ROOT_DIR, "data", "wsj02to21.linetrees")
COMBINED   = os.path.join(ROOT_DIR, "data", "combined.linetrees")

NS = "http://nite.sourceforge.net/"


# ─────────────────────────────────────────────────────────────
# STEP 1  Load terminal words (word + POS) for each conversation
# ─────────────────────────────────────────────────────────────
def load_terminals(term_dir):
    """Returns dict: local_id -> (word, pos)"""
    terminals = {}
    files = sorted(f for f in os.listdir(term_dir) if f.endswith(".xml"))
    print(f"Loading terminals from {len(files)} files...")

    for fname in files:
        fpath = os.path.join(term_dir, fname)
        base  = fname.replace(".terminals.xml", "")  # e.g. sw2005.A
        try:
            tree = etree.parse(fpath)
            root = tree.getroot()
            for elem in root:
                local = elem.tag.split("}")[-1]
                if local != "word":
                    continue
                local_id = elem.get(f"{{{NS}}}id", "")
                orth     = elem.get("orth", "").strip()
                pos      = elem.get("pos", "UNK")
                if orth and orth != "*":
                    # key = base.local_id e.g. sw2005.A.s1_1
                    terminals[f"{base}.{local_id}"] = (orth, pos)
        except Exception:
            continue

    print(f"  Loaded {len(terminals):,} terminal words")
    return terminals


# ─────────────────────────────────────────────────────────────
# STEP 2  Convert one syntax XML file to linetrees
# ─────────────────────────────────────────────────────────────
def extract_id_from_href(href, base):
    """
    'sw2005.A.terminals.xml#id(s1_1)' -> 'sw2005.A.s1_1'
    """
    m = re.search(r"#id\(([^)]+)\)", href)
    if not m:
        return None
    local_id = m.group(1)
    # Extract conv.speaker from href filename
    file_part = href.split("#")[0].replace(".terminals.xml", "")
    return f"{file_part}.{local_id}"


def node_to_linetree(elem, terminals, base):
    """
    Recursively convert an <nt> element to linetree string.
    Returns string like (NP (DT the) (NN board)) or None if empty.
    """
    local = elem.tag.split("}")[-1]

    if local == "child":
        # Leaf node — look up word and POS from terminals
        href   = elem.get("href", "")
        gid    = extract_id_from_href(href, base)
        if gid and gid in terminals:
            word, pos = terminals[gid]
            # Clean word for tree format
            word = word.replace("(", "-LRB-").replace(")", "-RRB-")
            pos  = pos.replace("(", "-LRB-").replace(")", "-RRB-")
            return f"({pos} {word})"
        return None

    if local == "nt":
        cat = elem.get("cat", "X")
        # Remove function tags
        cat = cat.split("-")[0].split("=")[0]
        if not cat:
            cat = "X"

        children_str = []
        for child in elem:
            child_local = child.tag.split("}")[-1]
            if child_local in ("child", "nt"):
                result = node_to_linetree(child, terminals, base)
                if result:
                    children_str.append(result)

        if not children_str:
            return None

        return f"({cat} {' '.join(children_str)})"

    return None


def convert_syntax_file(fpath, terminals):
    """Convert one syntax XML file to list of linetree strings."""
    base = os.path.basename(fpath).replace(".syntax.xml", "")
    trees = []

    try:
        tree = etree.parse(fpath)
        root = tree.getroot()

        for parse in root:
            local = parse.tag.split("}")[-1]
            if local != "parse":
                continue

            # Each <parse> has one root <nt>
            for child in parse:
                child_local = child.tag.split("}")[-1]
                if child_local == "nt":
                    result = node_to_linetree(child, terminals, base)
                    if result and len(result) > 10:
                        trees.append(result)
                    break

    except Exception as e:
        pass

    return trees


# ─────────────────────────────────────────────────────────────
# STEP 3  Process all syntax files
# ─────────────────────────────────────────────────────────────
def convert_all_syntax(syntax_dir, terminals):
    files    = sorted(f for f in os.listdir(syntax_dir)
                      if f.endswith(".xml"))
    total    = len(files)
    all_trees = []

    print(f"\nConverting {total} syntax files to linetrees...")

    for idx, fname in enumerate(files):
        fpath = os.path.join(syntax_dir, fname)
        trees = convert_syntax_file(fpath, terminals)
        all_trees.extend(trees)

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{total} done... ({len(all_trees)} trees so far)")

    print(f"\nTotal Switchboard trees extracted: {len(all_trees):,}")
    return all_trees


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("Convert Switchboard Syntax XML -> Linetrees")
    print("=" * 55)

    # 1. Load terminals
    terminals = load_terminals(TERM_DIR)

    # 2. Convert all syntax files
    swbd_trees = convert_all_syntax(SYNTAX_DIR, terminals)

    # 3. Save Switchboard linetrees
    with open(SWBD_LINES, "w") as f:
        for tree in swbd_trees:
            f.write(tree + "\n")
    print(f"\nSaved Switchboard linetrees -> {SWBD_LINES}")

    # 4. Combine with NLTK PTB linetrees
    if os.path.exists(NLTK_LINES):
        with open(NLTK_LINES, "r") as f:
            nltk_trees = [l.strip() for l in f if l.strip()]
        print(f"NLTK PTB trees: {len(nltk_trees):,}")
        print(f"Switchboard trees: {len(swbd_trees):,}")

        all_trees = nltk_trees + swbd_trees
        with open(COMBINED, "w") as f:
            for tree in all_trees:
                f.write(tree + "\n")
        print(f"Combined total: {len(all_trees):,} trees")
        print(f"Saved combined linetrees -> {COMBINED}")
    else:
        print(f"NLTK linetrees not found at {NLTK_LINES}")
        print("Run this first in WSL:")
        print("  cp ~/modelblocks-release/genmodel/wsj02to21.linetrees data/")

    print("\n" + "=" * 55)
    print("Conversion COMPLETE")
    print("=" * 55)
    print("\nNext step: run src/03_syntactic_surprisal.py")