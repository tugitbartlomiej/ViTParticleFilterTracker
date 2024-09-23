import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_predictions(image, predictions, threshold=0.7):
    plt.figure(figsize=(10, 10))
    plt.imshow(image.permute(1, 2, 0))
    ax = plt.gca()

    for score, label, box in zip(predictions['scores'], predictions['labels'], predictions['boxes']):
        if score > threshold:
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(box[0], box[1], f'{label}: {score:.2f}', bbox=dict(facecolor='white', alpha=0.8))

    plt.axis('off')
    plt.show()

# W funkcji evaluate_model dodaj:
for i in range(min(5, len(batch['pixel_values']))):
    image = batch['pixel_values'][i]
    outputs = model(pixel_values=image.unsqueeze(0).to(device))
    predictions = processor.post_process_object_detection(outputs, threshold=0.7)[0]
    visualize_predictions(image.cpu(), predictions)
