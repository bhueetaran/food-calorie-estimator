import json
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

# ---------- Load model ----------
model = MobileNetV2(weights="imagenet")

# ---------- Load calories ----------
with open("calories.json", "r") as f:
    CALS = json.load(f)

# Common synonym mapping (ImageNet label -> our calories key)
SYN_MAP = {
    "cheeseburger": "cheeseburger",
    "hamburger": "hamburger",
    "hotdog": "hotdog",
    "hot_dog": "hotdog",
    "french_loaf": "bagel",      # fallback to bread-like serving
    "bagel": "bagel",
    "pretzel": "pretzel",
    "pizza": "pizza",
    "burrito": "burrito",
    "taco": "taco",
    "ice_cream": "ice_cream",
    "ice_lolly": "ice_lolly",    # popsicle
    "chocolate_sauce": "chocolate_sauce",
    "guacamole": "guacamole",
    "sushi": "sushi",
    "carbonara": "spaghetti_carbonara",
    "spaghetti_squash": "spaghetti_bolognese",  # approx pasta fallback
    "ramen": "ramen",
    "paella": "paella",
    "omelet": "omelet",
    "cauliflower": "cauliflower",
    "broccoli": "broccoli",
    "banana": "banana",
    "orange": "orange",
    "lemon": "lemon",
    "strawberry": "strawberry",
    "pineapple": "pineapple",
    "granny_smith": "granny_smith",
    "pomegranate": "pomegranate",
    "French_toast": "pancake",   # breakfast fallback
    "waffle_iron": "waffle",     # rare label; fallback
    "waffle": "waffle",
    "panettone": "donut",        # sweet bakery fallback
    "dough": "donut",
    "doughnut": "donut",
    "cup": "ice_cream",          # if cup detected with dessert image, just a fallback
    "soup_bowl": "ramen",
    "plate": "sandwich",
    "meat_loaf": "fried_chicken",
    "submarine_sandwich": "sandwich",
    "croquette": "fried_chicken",
    "French_loaf": "bagel",
    "baguette": "bagel",
    "pizza_cutter": "pizza"
}

def pretty(name: str) -> str:
    return name.replace("_", " ").title()

def find_calorie_label(imagenet_label: str):
    """
    Try to map the predicted ImageNet label to our calories dictionary key.
    1) direct match
    2) synonym map
    3) simplified underscore/space variants
    """
    if imagenet_label in CALS:
        return imagenet_label

    if imagenet_label in SYN_MAP:
        mapped = SYN_MAP[imagenet_label]
        if mapped in CALS:
            return mapped

    # lowercase + underscore normalization attempts
    k = imagenet_label.lower().replace(" ", "_")
    if k in CALS:
        return k

    # try dropping qualifiers (e.g., 'granny_smith' already covered, but safe)
    simple = k.split(",")[0]
    if simple in CALS:
        return simple

    return None

def classify_image_bgr(bgr_image: np.ndarray):
    """Classify a BGR image (OpenCV frame). Returns (top5 list of (name, prob), chosen_label_for_cal)."""
    # Convert BGR->RGB then PIL
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb).resize((224, 224))
    x = np.array(pil_img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    top5 = decode_predictions(preds, top=5)[0]  # list of tuples: (wnid, label, prob)

    # Try to pick first that we can map to calories
    chosen_key = None
    for _, lbl, _ in top5:
        maybe = find_calorie_label(lbl)
        if maybe:
            chosen_key = maybe
            break

    # If none mapped, still return with top1 for display
    return [(lbl, float(prob)) for (_, lbl, prob) in top5], chosen_key

def main():
    print("== Food Recognition & Calories ==")
    print("Controls: [SPACE] capture & classify   [ESC] quit")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW helps on Windows
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: Cannot read frame.")
                break

            # Overlay helper text
            overlay = frame.copy()
            cv2.putText(overlay, "Press SPACE to capture & classify | ESC to exit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Webcam - Food Classifier", overlay)

            key = cv2.waitKey(10) & 0xFF
            if key == 27:  # ESC
                break
            if key == 32:  # SPACE
                # Classify current frame
                print("\nCapturing frame… classifying…")
                top5, cal_key = classify_image_bgr(frame)

                # Print top-5
                print("Top-5 predictions:")
                for i, (lbl, prob) in enumerate(top5, start=1):
                    print(f"  {i}. {pretty(lbl)} — {prob*100:.1f}%")

                # Calories
                if cal_key is not None:
                    food_name = pretty(cal_key)
                    kcal = CALS[cal_key]
                    print(f"→ Detected food for calories: {food_name}")
                    print(f"≈ Calories per serving: {kcal} kcal")
                else:
                    print("⚠ Couldn’t map prediction to a calories entry.")
                    print("  Tip: Capture a closer, clearer image of a single food item.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
