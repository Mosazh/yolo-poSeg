import os
import cv2
import yaml
import argparse
import numpy as np
import logging

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def find_image_file(label_file, image_dir):
    """
    Find the corresponding image file in image_dir based on the label file's basename (supports jpg/png/jpeg).
    """
    base = os.path.splitext(os.path.basename(label_file))[0]
    for ext in ['.jpg', '.png', '.jpeg']:
        candidate = os.path.join(image_dir, base + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def process_label_file(label_file, image_dir, vis_dir, kpt_num, kpt_dim):
    """
    Process a single label file:
      - Load the corresponding image.
      - For each line in the label file (if the first token is '0'), parse the keypoints and mask coordinates, and draw them on the image.
      - Save the visualization result to vis_dir.
    """
    image_file = find_image_file(label_file, image_dir)
    if image_file is None:
        msg = f"Image corresponding to {label_file} not found in {image_dir}"
        print(msg)
        logging.error(msg)
        return
    image = cv2.imread(image_file)
    if image is None:
        msg = f"Failed to load image: {image_file}"
        print(msg)
        logging.error(msg)
        return
    h, w = image.shape[:2]
    vis_image = image.copy()
    modified = False  # Indicates if any object is drawn

    with open(label_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        class_id = tokens[0]
        # Only perform keypoint check and visualization for class 0 (person)
        if class_id == '0':
            expected_kpt_tokens = kpt_num * kpt_dim  # e.g., 17*3 = 51
            if len(tokens) < 1 + expected_kpt_tokens:
                error_msg = (f"Error: In {label_file}, class 0 label does not have enough keypoint numbers. "
                             f"Expected {expected_kpt_tokens}, got {len(tokens)-1}.")
                print(error_msg)
                logging.error(error_msg)
                continue

            # Extract keypoints
            kpt_tokens = tokens[1:1+expected_kpt_tokens]
            keypoints = []
            for i in range(kpt_num):
                try:
                    x = float(kpt_tokens[i*3])
                    y = float(kpt_tokens[i*3+1])
                    vis_val = float(kpt_tokens[i*3+2])
                    keypoints.append((x, y, vis_val))
                except Exception as e:
                    error_msg = f"Error parsing keypoints in {label_file}: {e}"
                    print(error_msg)
                    logging.error(error_msg)
                    keypoints = []
                    break
            if len(keypoints) != kpt_num:
                error_msg = (f"Error: In {label_file}, the number of keypoints for class 0 is incorrect. "
                             f"Expected {kpt_num}, got {len(keypoints)}.")
                print(error_msg)
                logging.error(error_msg)
                continue

            # Check mask coordinates (all numbers after keypoints should be paired)
            mask_tokens = tokens[1+expected_kpt_tokens:]
            mask_points = []
            if mask_tokens:
                if len(mask_tokens) % 2 != 0:
                    error_msg = f"Error: In {label_file}, the number of mask coordinate numbers is not even."
                    print(error_msg)
                    logging.error(error_msg)
                else:
                    for i in range(0, len(mask_tokens), 2):
                        try:
                            mx = float(mask_tokens[i])
                            my = float(mask_tokens[i+1])
                            mask_points.append((mx, my))
                        except Exception as e:
                            error_msg = f"Error parsing mask coordinates in {label_file}: {e}"
                            print(error_msg)
                            logging.error(error_msg)
                            mask_points = []
                            break

            # Draw keypoints (red circles) and label indices (blue text) on the image
            for (x, y, vis_val) in keypoints:
                cx = int(x * w)
                cy = int(y * h)
                cv2.circle(vis_image, (cx, cy), 3, (0, 0, 255), -1)
            for idx, (x, y, vis_val) in enumerate(keypoints):
                cx = int(x * w)
                cy = int(y * h)
                cv2.putText(vis_image, str(idx), (cx+5, cy+5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # Draw mask polygon (green outline)
            if len(mask_points) >= 3:
                pts = []
                for (mx, my) in mask_points:
                    pts.append([int(mx * w), int(my * h)])
                pts = np.array(pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            modified = True
        else:
            # For non-class 0 labels, skip keypoint checking (add additional checks if needed)
            continue

    # If at least one object is drawn, save the visualization image
    if modified:
        vis_filename = os.path.join(vis_dir, os.path.basename(image_file))
        cv2.imwrite(vis_filename, vis_image)
        print(f"Visualization image saved: {vis_filename}")

def process_dataset(labels_dir, images_dir, vis_dir, kpt_num, kpt_dim):
    """
    Iterate through all .txt label files in labels_dir and process them.
    """
    for root, _, files in os.walk(labels_dir):
        for file in files:
            if file.endswith(".txt"):
                label_file = os.path.join(root, file)
                process_label_file(label_file, images_dir, vis_dir, kpt_num, kpt_dim)

def main():
    parser = argparse.ArgumentParser(
        description="Validate dataset labels and visualize keypoints and segmentation masks, while logging errors to a log file."
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the dataset configuration file")
    args = parser.parse_args()

    # Load YAML configuration file
    config = load_config(args.config)
    dataset_path = config.get("path", ".")
    # Set the log file path; logs will be saved in error.log under dataset_path
    log_file = os.path.join(dataset_path, "error.log")
    logging.basicConfig(filename=log_file,
                        level=logging.ERROR,
                        format="%(asctime)s:%(levelname)s:%(message)s")

    # Get training and validation image directories (relative to dataset_path)
    train_dir = os.path.join(dataset_path, config.get("train", ""))
    val_dir = os.path.join(dataset_path, config.get("val", ""))
    # Label directories: assumed to be at path/labels/train and path/labels/val respectively
    labels_train_dir = os.path.join(dataset_path, "labels", "train")
    labels_val_dir = os.path.join(dataset_path, "labels", "val")
    # Directory for saving visualization results
    vis_dir = os.path.join(dataset_path, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    # Get keypoint configuration, e.g., kpt_shape: [17, 3]
    kpt_shape = config.get("kpt_shape", [17, 3])
    kpt_num = kpt_shape[0]
    kpt_dim = kpt_shape[1]

    print("Start processing training label files...")
    process_dataset(labels_train_dir, train_dir, vis_dir, kpt_num, kpt_dim)
    print("Start processing validation label files...")
    process_dataset(labels_val_dir, val_dir, vis_dir, kpt_num, kpt_dim)
    print("Processing complete!")

if __name__ == "__main__":
    main()
