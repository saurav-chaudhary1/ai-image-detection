import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from datetime import datetime

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VIT_CHECKPOINT_8k = "ExperimentResults/checkpoints/ViT_best.pth"
EFF_CHECKPOINT_8k = "ExperimentResults/checkpoints/EfficientNet_best.pth"

VIT_CHECKPOINT_25k = "ResultFinal25k/checkpoints/ViT_best.pth"
EFF_CHECKPOINT_25k = "ResultFinal25k/checkpoints/EfficientNet_best.pth"


IMAGE_DATA = {
    "TestImages/test1.jpg": "REAL",
    "TestImages/test2.jpg": "REAL",
    "TestImages/test3.jpg": "FAKE",
    "TestImages/test4.jpg": "FAKE",
    "TestImages/test5.jpg": "FAKE",
    "TestImages/test6.jpg": "FAKE",
    "TestImages/test7.jpg": "REAL",
    "TestImages/test8.jpg": "FAKE",
}


def get_vit_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

def get_eff_transform():
    return transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.CenterCrop(240),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def load_vit(checkpoint_path):
    model = models.vit_b_16(weights=None)
    model.heads = nn.Linear(768, 1)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(DEVICE)
    model.eval()
    return model

def load_efficientnet(checkpoint_path):
    model = models.efficientnet_b1(weights=None)
    model.classifier[1] = nn.Linear(1280, 1)

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(DEVICE)
    model.eval()
    return model

def predict(model, image, transform):
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        prob = torch.sigmoid(output).item()

    label = "FAKE" if prob >= 0.5 else "REAL"
    confidence = prob if prob >= 0.5 else (1 - prob)

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "prob": round(prob, 4)
    }
    
def save_to_log(results, filename="results_log.txt"):
    with open(filename, "w") as f:
        f.write("Deepfake Detection Results Log\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated at: {datetime.now()}\n\n")

        for img_path, data in results.items():
            f.write(f"Image: {img_path}\n")
            f.write(f"Ground Truth: {data['ground_truth']}\n")

            for model_name, pred in data["predictions"].items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  Prediction : {pred['label']}\n")
                f.write(f"  Confidence : {pred['confidence']}\n")
                f.write(f"  Prob       : {pred['prob']}\n")

            f.write("\n" + "-" * 50 + "\n\n")

    print(f"Log file saved → {filename}")


def main():
    print(f"Device: {DEVICE}\n")

    print("Loading models...\n")

    models_dict = {
        "ViT_8k": load_vit(VIT_CHECKPOINT_8k),
        "Eff_8k": load_efficientnet(EFF_CHECKPOINT_8k),
        "ViT_25k": load_vit(VIT_CHECKPOINT_25k),
        "Eff_25k": load_efficientnet(EFF_CHECKPOINT_25k),
    }

    results = {}

    for i, (img_path, gt_label) in enumerate(IMAGE_DATA.items(), 1):
        print(f"\n{'='*50}")
        print(f"Image {i}: {img_path}")
        print(f"Ground Truth: {gt_label}")

        image = Image.open(img_path).convert("RGB")

        results[img_path] = {
            "ground_truth": gt_label,
            "predictions": {}
        }

        for name, model in models_dict.items():
            if "ViT" in name:
                pred = predict(model, image, get_vit_transform())
            else:
                pred = predict(model, image, get_eff_transform())

            results[img_path]["predictions"][name] = pred

            print(f"\n{name}:")
            print(f"  Prediction : {pred['label']}")
            print(f"  Confidence : {pred['confidence']}")
            print(f"  Prob       : {pred['prob']}")

    return results


if __name__ == "__main__":
    final_results = main()
    save_to_log(final_results, "comparison_results.txt")