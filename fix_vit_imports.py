import os
import re

vit_root = os.path.join("src", "vit")

pattern1 = re.compile(r'^(from|import)\s+vit\.', re.MULTILINE)
pattern2 = re.compile(r'FaceMorphDetection\.Src\.', re.IGNORECASE)

changed = []

for dirpath, _, filenames in os.walk(vit_root):
    for fname in filenames:
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(dirpath, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        new_content = pattern1.sub(lambda m: f"{m.group(1)} src.vit.", content)
        new_content = pattern2.sub("src.", new_content)

        if new_content != content:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(new_content)
            changed.append(fpath)

print(f"Patched {len(changed)} files:")
for f in changed:
    print(" ", f)