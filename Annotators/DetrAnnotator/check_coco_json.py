import json
from collections import defaultdict


def check_coco_json(json_path):
    print(f"Analiza pliku COCO: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    print(f"Liczba obrazów: {len(images)}")
    print(f"Liczba adnotacji: {len(annotations)}")
    print(f"Liczba kategorii: {len(categories)}")

    # Mapowanie image_id -> adnotacje
    anns_per_image = defaultdict(list)
    for ann in annotations:
        anns_per_image[ann["image_id"]].append(ann)

    images_with_anns = 0
    images_without_anns = 0
    multi_anns = 0
    for img in images:
        img_id = img.get("id")
        anns = anns_per_image.get(img_id, [])
        if len(anns) == 0:
            images_without_anns += 1
        else:
            images_with_anns += 1
            if len(anns) > 1:
                multi_anns += 1

    print(f"Obrazy z adnotacjami: {images_with_anns}")
    print(f"Obrazy bez adnotacji: {images_without_anns}")
    print(f"Obrazy z wieloma adnotacjami: {multi_anns}")

    # Przykładowe obrazy bez adnotacji
    if images_without_anns > 0:
        print("\nPrzykładowe obrazy bez adnotacji:")
        count = 0
        for img in images:
            img_id = img.get("id")
            if len(anns_per_image.get(img_id, [])) == 0:
                print(f"  id: {img_id}, file_name: {img.get('file_name', img.get('aug_source', 'brak'))}")
                count += 1
                if count >= 10:
                    break

    # Przykładowe adnotacje
    if len(annotations) > 0:
        print("\nPrzykładowe adnotacje:")
        for ann in annotations[:5]:
            print(f"  image_id: {ann['image_id']}, bbox: {ann['bbox']}, category_id: {ann['category_id']}")

    # Statystyka: ile adnotacji na obraz
    anns_count = [len(anns_per_image[img.get("id")]) for img in images]
    if anns_count:
        avg_anns = sum(anns_count) / len(anns_count)
        print(f"\nŚrednia liczba adnotacji na obraz: {avg_anns:.2f}")

    print("\nAnaliza zakończona.")

if __name__ == "__main__":
    # Podaj ścieżkę do pliku COCO JSON tutaj:
    json_path = "F:/Studia/PhD_projekt/VIT/ViTParticleFilterTracker/Annotators/DetrAnnotator/augmented_dataset/augmented_coco_450-14660_20250417_133558.json"
    check_coco_json(json_path)