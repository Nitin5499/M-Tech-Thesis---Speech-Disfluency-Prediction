"""
map_pipeline_io.py
Scans every .py and .R script in src/ and lists which data files each one
reads and which it writes. Output: pipeline_io_map.csv and pipeline_io_map.md
(in the repo root) plus a short summary printed on screen.

Run from the repo root:
    python src/map_pipeline_io.py
Optional:
    python src/map_pipeline_io.py --src src --out .
It only reads the scripts, it does not run them or touch any data.
"""
import argparse
import csv
import os
import re
import sys

# file extensions we care about
EXT = r"(?:csv|tsv|txt|rds|RData|rda|json|pkl|pickle|npy|npz|parquet|png|pdf|tex|xml|dict|bin|pt|log)"

# quoted strings that end in one of those extensions
STR_PATH = re.compile(r"""(['"])([^'"\n]*?\.""" + EXT + r""")\1""")

# calls that clearly READ a file
READ_HINT = re.compile(
    r"(read_csv|read\.csv|read_table|read_delim|fread|readRDS|load\(|"
    r"pd\.read_|np\.load|json\.load|pickle\.load|open\([^)]*['\"]r|read_parquet|"
    r"read_xml|ET\.parse|readLines|scan\()"
)
# calls that clearly WRITE a file
WRITE_HINT = re.compile(
    r"(to_csv|write\.csv|write_csv|fwrite|saveRDS|save\(|np\.save|json\.dump|"
    r"pickle\.dump|savefig|ggsave|png\(|pdf\(|open\([^)]*['\"][wa]|"
    r"to_parquet|writeLines|sink\(|\.write\(|to_latex|to_json)"
)

# variables holding paths, e.g. OUT = "data/processed/x.csv"
ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:<-|=)\s*(.*)$")


def scan_script(path):
    """Return list of (kind, filename, line_no) for one script."""
    found = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        print("could not read", path, e)
        return found

    for i, line in enumerate(lines, start=1):
        code = line.split("#")[0] if not line.lstrip().startswith(("'", '"')) else line
        paths = [m.group(2) for m in STR_PATH.finditer(code)]
        if not paths:
            continue
        # look at this line and the one before (calls often wrap)
        # use this line first; only look at the previous line if this one has no hint
        is_w = bool(WRITE_HINT.search(code))
        is_r = bool(READ_HINT.search(code))
        if not is_w and not is_r and i > 1:
            prev = lines[i - 2]
            is_w = bool(WRITE_HINT.search(prev))
            is_r = bool(READ_HINT.search(prev))
        for p in paths:
            if is_w and not is_r:
                kind = "WRITE"
            elif is_r and not is_w:
                kind = "READ"
            elif is_r and is_w:
                kind = "READ+WRITE?"
            else:
                kind = "PATH?"  # a path string that is just assigned to a variable
            found.append((kind, p, i))
    return found


def stage_key(name):
    """Sort by the leading number in the filename (02_, 14_, ...), then name."""
    m = re.match(r"(\d+)", name)
    return (int(m.group(1)) if m else 9999, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="src")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        sys.exit("folder '%s' not found. Run this from the repo root." % args.src)

    scripts = []
    for root, _, files in os.walk(args.src):
        for fn in files:
            if fn.endswith((".py", ".R", ".r")):
                scripts.append(os.path.join(root, fn))
    scripts.sort(key=lambda p: stage_key(os.path.basename(p)))

    rows = []
    writers = {}  # file -> scripts that write it
    readers = {}  # file -> scripts that read it
    for sp in scripts:
        name = os.path.relpath(sp, args.src)
        for kind, fpath, ln in scan_script(sp):
            rows.append([name, kind, fpath, ln])
            base = os.path.basename(fpath)
            if kind.startswith("WRITE") or kind == "READ+WRITE?":
                writers.setdefault(base, set()).add(name)
            if kind.startswith("READ"):
                readers.setdefault(base, set()).add(name)

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "pipeline_io_map.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["script", "kind", "file", "line"])
        w.writerows(rows)

    # which scripts feed which (a script that reads a file another one writes)
    deps = {}
    for base, rs in readers.items():
        for r in rs:
            for wr in writers.get(base, []):
                if wr != r:
                    deps.setdefault(r, set()).add((wr, base))

    md_path = os.path.join(args.out, "pipeline_io_map.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Pipeline I/O map\n\n")
        f.write("Scripts scanned: %d\n\n" % len(scripts))
        f.write("## Per script\n\n")
        for sp in scripts:
            name = os.path.relpath(sp, args.src)
            mine = [r for r in rows if r[0] == name]
            f.write("### %s\n" % name)
            if not mine:
                f.write("- no data file paths found\n\n")
                continue
            for _, kind, fp, ln in mine:
                f.write("- %s `%s` (line %d)\n" % (kind, fp, ln))
            f.write("\n")
        f.write("## Which script feeds which\n\n")
        for r in sorted(deps, key=stage_key):
            for wr, base in sorted(deps[r]):
                f.write("- %s  <-  %s  (via `%s`)\n" % (r, wr, base))
        f.write("\n## Files read but never written by any script (raw inputs)\n\n")
        for base in sorted(readers):
            if base not in writers:
                f.write("- `%s` (read by %s)\n" % (base, ", ".join(sorted(readers[base]))))
        f.write("\n## Files written but never read (final outputs or dead ends)\n\n")
        for base in sorted(writers):
            if base not in readers:
                f.write("- `%s` (written by %s)\n" % (base, ", ".join(sorted(writers[base]))))

    print("scripts scanned :", len(scripts))
    print("file references :", len(rows))
    print("raw inputs      :", sum(1 for b in readers if b not in writers))
    print("final/dead outs :", sum(1 for b in writers if b not in readers))
    print("wrote", csv_path)
    print("wrote", md_path)


if __name__ == "__main__":
    main()