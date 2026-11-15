"""
repo_inventory.py
Run from the repo root:   python repo_inventory.py
Prints (and saves to repo_inventory.txt) three things I need to build the
replication pipeline:
  1. every file under src/
  2. every file under data/processed/ with its size
  3. guesses for where the raw inputs live (NXT XML, OANC, ModelBlocks)
"""
import os
import sys

OUT = []

def say(line=""):
    print(line)
    OUT.append(line)

def list_tree(folder, show_size=False):
    if not os.path.isdir(folder):
        say(f"  (folder not found: {folder})")
        return
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for f in sorted(files):
            p = os.path.join(root, f)
            if show_size:
                mb = os.path.getsize(p) / 1e6
                say(f"  {p}  ({mb:.1f} MB)")
            else:
                say(f"  {p}")

# 1. src folder
say("=== 1. src/ ===")
list_tree("src")

# 2. processed data
say("")
say("=== 2. data/processed/ ===")
list_tree(os.path.join("data", "processed"), show_size=True)

# 3. raw inputs: look for folders whose name hints at the raw data
say("")
say("=== 3. possible raw-input locations ===")
KEYWORDS = ("switchboard", "nxt", "oanc", "modelblocks", "model-blocks", "berkeley")
SKIP = {"windows", "program files", "program files (x86)", "appdata",
        "node_modules", ".git", "$recycle.bin", "programdata", "site-packages"}

# start in the repo's parent and the user's home; add drives if you keep data elsewhere
starts = {os.path.abspath(os.path.join(os.getcwd(), "..")), os.path.expanduser("~")}
extra = sys.argv[1:]          # optional: python repo_inventory.py D:\ E:\data
starts.update(extra)

seen = set()
for start in starts:
    for root, dirs, files in os.walk(start):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP]
        for d in dirs:
            if any(k in d.lower() for k in KEYWORDS):
                p = os.path.join(root, d)
                if p in seen:
                    continue
                seen.add(p)
                n_xml = 0
                try:
                    for _, _, fs in os.walk(p):
                        n_xml += sum(1 for x in fs if x.lower().endswith(".xml"))
                except OSError:
                    pass
                say(f"  {p}   [{n_xml} xml files inside]")

if not seen:
    say("  nothing found - rerun with the drive where the data is, e.g.")
    say("  python repo_inventory.py D:\\")

with open("repo_inventory.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT))
say("")
say("saved to repo_inventory.txt - paste that file here")