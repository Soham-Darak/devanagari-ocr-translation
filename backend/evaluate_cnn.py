from pathlib import Path
from PIL import Image
from app.services.model_adapter import CharacterClassifier

dataset = Path("../DevanagariHandwrittenCharacterDataset/Test")
checkpoint = Path("models/devanagari_cnn_best.pth")
extensions = {".png", ".jpg", ".jpeg", ".bmp"}

classifier = CharacterClassifier(str(checkpoint))
classifier.load()

total = 0
correct = 0
class_results = []

for class_dir in sorted(dataset.iterdir()):
    if not class_dir.is_dir():
        continue

    class_total = 0
    class_correct = 0

    for image_path in class_dir.iterdir():
        if image_path.suffix.lower() not in extensions:
            continue

        with Image.open(image_path) as image:
            result = classifier.predict_character(image.convert("L"), top_k=1)

        total += 1
        class_total += 1

        if result["predicted_class"] == class_dir.name:
            correct += 1
            class_correct += 1

    if class_total:
        class_results.append(
            (class_dir.name, class_correct, class_total)
        )

print("\n--- Full Test Dataset Results ---")
print(f"Total images: {total}")
print(f"Correct predictions: {correct}")
print(f"Incorrect predictions: {total - correct}")
print(f"Accuracy: {100 * correct / total:.2f}%" if total else "No images found.")

print("\n--- Lowest-accuracy classes ---")
for name, good, count in sorted(
    class_results, key=lambda row: row[1] / row[2]
)[:10]:
    print(f"{name}: {good}/{count} ({100 * good / count:.2f}%)")
