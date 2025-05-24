import os
import json
import gc
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

class SlidingWindowPredictor:
    def __init__(self, model_path, window_size=1280, overlap=0.2, device='cpu', save_window_vis=False, scale_factor=1.0):
        """
        Args:
            model_path (str): Path to the YOLO model .pt file.
            window_size (int): Size of each sliding window (square).
            overlap (float): Fractional overlap between windows (0 to <1).
            device (str): 'cpu' or 'cuda:0' for inference device.
            save_window_vis (bool): Whether to save each window's visualization.
            scale_factor (float): Scale factor for input image (e.g., 0.5 to halve size).
        """
        self.model = YOLO(model_path)
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))
        self.device = device
        self.save_window_vis = save_window_vis
        self.scale_factor = scale_factor

    def _sliding_windows(self, img_shape):
        """Generate (x1, y1, x2, y2) for each window over the image."""
        h, w = img_shape[:2]
        windows = []
        for y in range(0, h, self.stride):
            for x in range(0, w, self.stride):
                x_end = min(x + self.window_size, w)
                y_end = min(y + self.window_size, h)
                windows.append((x, y, x_end, y_end))
        return windows

    def predict_and_save(self, image_path, output_dir):
        """
        Run sliding-window inference and save separate outputs:
        - class 0 detections to CSV with slice, bbox, confidence, class_id, keypoints
        - class 1 masks as convex hulls in JSON
        Returns:
            merged_img (np.ndarray): Final blended image array.
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Records for class 0 and class 1
        class0_records = []
        class1_records = []

        # Read and scale image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        orig_h, orig_w = img.shape[:2]
        if self.scale_factor != 1.0:
            new_h, new_w = int(orig_h * self.scale_factor), int(orig_w * self.scale_factor)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        full_h, full_w = img.shape[:2]

        # Prepare blending buffers
        merged_sum = np.zeros((full_h, full_w, 3), dtype=np.float32)
        weight_sum = np.zeros((full_h, full_w), dtype=np.float32)

        windows = self._sliding_windows(img.shape)
        for idx, (x1, y1, x2, y2) in enumerate(tqdm(windows, desc="Sliding Windows")):
            # Extract crop
            crop = img[y1:y2, x1:x2]
            crop_h, crop_w = crop.shape[:2]

            # Create a square input with padding to maintain 1:1 aspect ratio
            max_dim = max(crop_h, crop_w)
            square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)

            try:
                # Center the crop in the square image
                y_offset = (max_dim - crop_h) // 2
                x_offset = (max_dim - crop_w) // 2

                # Ensure valid dimensions
                if crop_h <= 0 or crop_w <= 0:
                    print(f"Warning: Invalid crop dimensions at position ({x1}, {y1}): {crop_w}x{crop_h}")
                    continue

                square_img[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w] = crop

            except ValueError as e:
                print(f"Warning: Error processing crop at position ({x1}, {y1}): {str(e)}")
                continue

            # Run prediction on the square image
            try:
                results = self.model.predict(square_img, imgsz=self.window_size, show_labels=False, conf=0.5, device=self.device)
            except Exception as e:
                print(f"Error during prediction at position ({x1}, {y1}): {str(e)}")
                continue
            res = results[0]

            # Extract masks if available
            masks = None
            if hasattr(res, 'masks') and res.masks is not None:
                masks = res.masks.data.cpu().numpy().astype(np.uint8)

            # Extract keypoints if available
            keypoints = None
            if hasattr(res, 'keypoints') and res.keypoints is not None:
                keypoints = res.keypoints.data.cpu().numpy()  # shape: (n, K, 3)

            # Calculate offsets for the padded square image
            crop_h, crop_w = crop.shape[:2]
            max_dim = max(crop_h, crop_w)
            y_offset = (max_dim - crop_h) // 2
            x_offset = (max_dim - crop_w) // 2

            # Process detections
            for i, box in enumerate(res.boxes):
                cls_idx = int(box.cls.item())
                x1b, y1b, x2b, y2b = box.xyxy[0].tolist()
                conf = float(box.conf.item())

                # Adjust coordinates for padding offset
                x1b -= x_offset
                y1b -= y_offset
                x2b -= x_offset
                y2b -= y_offset

                # Clip to ensure coordinates are within the original crop
                x1b = max(0, min(x1b, crop_w))
                y1b = max(0, min(y1b, crop_h))
                x2b = max(0, min(x2b, crop_w))
                y2b = max(0, min(y2b, crop_h))

                # Skip if box is completely outside the original crop
                if x1b >= crop_w or y1b >= crop_h or x2b <= 0 or y2b <= 0:
                    continue

                # Adjust coordinates for scaling
                scale = 1.0 / self.scale_factor
                x1b, y1b, x2b, y2b = [v * scale for v in [x1b, y1b, x2b, y2b]]

                if cls_idx == 0:
                    # Gather keypoints for this instance
                    kpts_str = ''
                    if keypoints is not None and i < keypoints.shape[0]:
                        try:
                            pts = []
                            for kp in keypoints[i]:
                                # Check if keypoint has visibility information
                                if len(kp) >= 3:
                                    kp_x, kp_y, kp_v = kp[:3]
                                    # Skip keypoints with low visibility
                                    if kp_v < 0.1:  # Threshold for visibility
                                        continue
                                else:
                                    kp_x, kp_y = kp[:2]

                                # Adjust keypoint coordinates for padding offset
                                kp_x = float(kp_x - x_offset)
                                kp_y = float(kp_y - y_offset)

                                # Skip keypoints outside the original crop
                                if kp_x < 0 or kp_x >= crop_w or kp_y < 0 or kp_y >= crop_h:
                                    continue

                                # Map back to original image coordinates
                                global_kp_x = kp_x * scale + x1 * scale
                                global_kp_y = kp_y * scale + y1 * scale

                                pts.append((global_kp_x, global_kp_y))

                            kpts_str = ';'.join([f"({x:.1f},{y:.1f})" for x, y in pts])
                        except Exception as e:
                            print(f"Warning: Error processing keypoints at position ({x1}, {y1}): {str(e)}")

                    # Map bounding box coordinates back to original image
                    global_x1b = x1b + x1 * scale
                    global_y1b = y1b + y1 * scale
                    global_x2b = x2b + x1 * scale
                    global_y2b = y2b + y1 * scale

                    class0_records.append({
                        'slice_x': x1 * scale,
                        'slice_y': y1 * scale,
                        'x_min': float(global_x1b),
                        'y_min': float(global_y1b),
                        'x_max': float(global_x2b),
                        'y_max': float(global_y2b),
                        'confidence': conf,
                        'class_id': cls_idx,
                        'keypoints': kpts_str
                    })

                elif cls_idx == 1 and masks is not None:
                    mask_i = masks[i]

                    # Adjust mask for padding (crop the relevant part)
                    try:
                        # Check mask dimensions
                        mask_h, mask_w = mask_i.shape[:2]

                        # If mask dimensions match the padded square
                        if mask_h == max_dim and mask_w == max_dim:
                            # Extract the portion corresponding to the original crop
                            mask_i = mask_i[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w]
                        # If mask dimensions don't match, resize to match the crop
                        elif mask_h != crop_h or mask_w != crop_w:
                            mask_i = cv2.resize(mask_i, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
                    except Exception as e:
                        print(f"Warning: Error processing mask at position ({x1}, {y1}): {str(e)}")
                        continue

                    # Resize mask to reduce memory usage
                    mask_i = cv2.resize(mask_i, (int(mask_i.shape[1] * scale), int(mask_i.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)

                    try:
                        # Find contours to get points for convex hull
                        contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            # Compute convex hull for the largest contour
                            contour = max(contours, key=cv2.contourArea)
                            hull = cv2.convexHull(contour)

                            # Convert hull points to list and adjust to global coordinates
                            hull_points = []
                            for pt in hull:
                                # Map contour point to original image coordinates
                                pt_x = int(pt[0][0] + x1 * scale)
                                pt_y = int(pt[0][1] + y1 * scale)
                                hull_points.append([pt_x, pt_y])

                            # Only add if we have valid hull points
                            if hull_points:
                                class1_records.append({
                                    'slice_x': x1 * scale,
                                    'slice_y': y1 * scale,
                                    'convex_hull': hull_points
                                })
                    except Exception as e:
                        print(f"Warning: Error processing mask contours at position ({x1}, {y1}): {str(e)}")

            # Optional: save window visualization
            if self.save_window_vis:
                vis = res.plot(labels=False,)
                vis_path = os.path.join(output_dir, f"{base_name}_win{idx}.jpg")
                cv2.imwrite(vis_path, vis)
                del vis  # Release visualization memory

            # Blend visualization
            try:
                vis_float = res.plot(labels=False,).astype(np.float32)

                # If visualization is from padded square image, extract original crop region
                if vis_float.shape[:2] == (max_dim, max_dim):
                    vis_float = vis_float[y_offset:y_offset+crop_h, x_offset:x_offset+crop_w]

                # Ensure visualization matches crop dimensions
                if vis_float.shape[:2] != (crop_h, crop_w):
                    vis_float = cv2.resize(vis_float, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

                merged_sum[y1:y2, x1:x2] += vis_float * 0.7
                weight_sum[y1:y2, x1:x2] += 0.7
            except Exception as e:
                print(f"Warning: Error processing visualization at position ({x1}, {y1}): {str(e)}")

            # Clean up
            del crop, results, res, masks, keypoints, vis_float
            gc.collect()

        # Final blended image
        weight_exp = np.expand_dims(weight_sum, axis=2)
        blended = np.divide(merged_sum, weight_exp, out=np.zeros_like(merged_sum), where=weight_exp > 0)
        merged_img = blended.astype(np.uint8)
        # Resize back to original size if scaled
        if self.scale_factor != 1.0:
            merged_img = cv2.resize(merged_img, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        merged_path = os.path.join(output_dir, f"{base_name}_merged.jpg")
        cv2.imwrite(merged_path, merged_img)

        # Save class 0 CSV
        df0 = pd.DataFrame(class0_records)
        csv0_path = os.path.join(output_dir, f"{base_name}_class0.csv")
        df0.to_csv(csv0_path, index=False)

        # Save class 1 convex hulls to JSON
        json1_path = os.path.join(output_dir, f"{base_name}_class1.json")
        with open(json1_path, 'w') as jf:
            json.dump(class1_records, jf, indent=2)

        print(f"Saved merged image: {merged_path}")
        print(f"Saved class0 CSV: {csv0_path} (rows: {len(df0)})")
        print(f"Saved class1 JSON: {json1_path} (objects: {len(class1_records)})")

        # Clean up
        del img, merged_sum, weight_sum, blended
        gc.collect()

        return merged_img

if __name__ == "__main__":
    predictor = SlidingWindowPredictor(
        model_path="/home/Mos/Documents/Complex/MyStudy/new_yolo/train_record/poseg/yolov8-poseg_ALSS_v2_Ciou/weights/best.pt",
        window_size=2560,
        overlap=0,  # Increased overlap to maintain coverage
        device='cpu',
        save_window_vis=False,  # Disabled to save memory
        scale_factor=0.5  # Scale image to 50% size
    )
    predictor.predict_and_save(
        image_path="/home/Mos/Desktop/mtemp/complete_test/Experimental_plot_02.png",
        output_dir="/home/Mos/Desktop/mtemp/complete_test/HNLYWD_plot2_pred/pred_v8_ALSS_v2_Ciou_window2560_overlap0_scale0.5"
    )

