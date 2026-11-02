from pathlib import Path
import shutil

DATASET = Path("dataset")
CATEGORIES = ["single_person", "multiple_people"]

for split in ["train", "validation", "test"]:
    for category in CATEGORIES:
        target = DATASET / split / category

        if not target.exists():
            print(f"Folder not found: {target}")
            continue

        # Find images in nested folders and move them up
        images = [
            f for f in target.rglob("*")
            if f.is_file()
            and f.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        ]

        moved = 0
        for image in images:
            destination = target / image.name

            if image.parent != target and not destination.exists():
                shutil.move(str(image), str(destination))
                moved += 1

        # Remove empty nested directories
        for folder in sorted(target.rglob("*"), reverse=True):
            if folder.is_dir():
                try:
                    folder.rmdir()
                except OSError:
                    pass

        print(f"{split}/{category}: moved {moved} images")

print("Dataset folder structure check completed.")