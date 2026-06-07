import os
from collections import defaultdict

extensions = {".jpg", ".jpeg", ".png"}
ignore_folders={"Site", "Sites"}

root_folder = "/path/to/input/folder/"

counts = defaultdict(int)

for top_folder in os.listdir(root_folder):
    top_path = os.path.join(root_folder, top_folder)

    if not os.path.isdir(top_path):
        continue

    for root, directories, files in os.walk(top_path):
        directories[:] = [d for d in directories if d not in ignore_folders]
        for file in files:
            extension = os.path.splitext(file)[1].lower()
            if extension in extensions:
                counts[top_folder] += 1

print(f"{'Folder':40} Images")
print("-" * 50)

for folder, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
    print(f"{folder:40} {count:>8}")

if counts:
    values = list(counts.values())

    print("\nStatistics")
    print("-" * 50)
    print(f"Number of classes : {len(values)}")
    print(f"Total images      : {sum(values)}")
    print(f"Largest class     : {max(values)}")
    print(f"Smallest class    : {min(values)}")
    print(f"Ratio max/min     : {max(values) / min(values):.2f}")