import os
import json
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

class SlidingWindowPredictor:
    def __init__(self, model_path, window_size=640, overlap=0.2, device='cpu', save_window_vis=False):
        """
        Args:
            model_path (str): Path to the YOLO model .pt file.
            window_size (int): Size of each sliding window (square).
            overlap (float): Fractional overlap between windows (0 to <1).
            device (str): 'cpu' or 'cuda:0' for inference device.
            save_window_vis (bool): Whether to save each window's visualization.
        """
        self.model = YOLO(model_path)
        self.window_size = window_size
        self.stride = int(window_size * (1 - overlap))
        self.device = device
        self.save_window_vis = save_window_vis

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
        - class 1 masks as contours in JSON
        Returns:
            merged_img (np.ndarray): Final blended image array.
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Records for class 0 and class 1
        class0_records = []
        class1_records = []

        # Read image
        img = cv2.imread(image_path)
        full_h, full_w = img.shape[:2]

        # Prepare blending buffers
        merged_sum = np.zeros_like(img, dtype=float)
        weight_sum = np.zeros((full_h, full_w), dtype=float)

        windows = self._sliding_windows(img.shape)
        for idx, (x1, y1, x2, y2) in enumerate(tqdm(windows, desc="Sliding Windows")):
            crop = img[y1:y2, x1:x2]
            results = self.model(crop, imgsz=self.window_size, conf=0.34, device=self.device, verbose=False)
            res = results[0]

            # Extract masks if available
            masks = None
            if hasattr(res, 'masks') and res.masks is not None:
                masks = res.masks.data.cpu().numpy().astype(np.uint8)

            # Extract keypoints if available
            keypoints = None
            if hasattr(res, 'keypoints') and res.keypoints is not None:
                keypoints = res.keypoints.data.cpu().numpy()  # shape: (n, K, 3)

            # Process detections
            for i, box in enumerate(res.boxes):
                cls_idx = int(box.cls.item())
                x1b, y1b, x2b, y2b = box.xyxy[0].tolist()
                conf = float(box.conf.item())

                if cls_idx == 0:
                    # Gather keypoints for this instance
                    kpts_str = ''
                    if keypoints is not None and i < keypoints.shape[0]:
                        pts = []
                        for kp in keypoints[i]:
                            pts.append((float(kp[0]), float(kp[1])))
                        kpts_str = ';'.join([f"{x:.1f}:{y:.1f}" for x, y in pts])
                    class0_records.append({
                        'slice_x': x1,
                        'slice_y': y1,
                        'x_min': float(x1b),
                        'y_min': float(y1b),
                        'x_max': float(x2b),
                        'y_max': float(y2b),
                        'confidence': conf,
                        'class_id': cls_idx,
                        'keypoints': kpts_str
                    })

                elif cls_idx == 1 and masks is not None:
                    mask_i = masks[i]
                    # Find contours and reshape for consistent format
                    contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    formatted = []
                    for cnt in contours:
                        pts = cnt.reshape(-1, 2)
                        pts_list = [[int(pt[0] + x1), int(pt[1] + y1)] for pt in pts]
                        formatted.append(pts_list)
                    class1_records.append({
                        'slice_x': x1,
                        'slice_y': y1,
                        'contours': formatted
                    })

            # Optional: save window visualization
            vis = res.plot()
            if self.save_window_vis:
                vis_path = os.path.join(output_dir, f"{base_name}_win{idx}.jpg")
                cv2.imwrite(vis_path, vis)

            # Blend visualization
            vis_float = vis.astype(float)
            merged_sum[y1:y2, x1:x2] += vis_float * 0.7
            weight_sum[y1:y2, x1:x2] += 0.7

        # Final blended image
        weight_exp = np.expand_dims(weight_sum, axis=2)
        blended = np.divide(merged_sum, weight_exp, out=np.zeros_like(merged_sum), where=weight_exp>0)
        merged_img = blended.astype(np.uint8)
        merged_path = os.path.join(output_dir, f"{base_name}_merged.jpg")
        cv2.imwrite(merged_path, merged_img)

        # Save class 0 CSV
        df0 = pd.DataFrame(class0_records)
        csv0_path = os.path.join(output_dir, f"{base_name}_class0.csv")
        df0.to_csv(csv0_path, index=False)

        # Save class 1 contours to JSON
        json1_path = os.path.join(output_dir, f"{base_name}_class1.json")
        with open(json1_path, 'w') as jf:
            json.dump(class1_records, jf, indent=2)

        print(f"Saved merged image: {merged_path}")
        print(f"Saved class0 CSV: {csv0_path} (rows: {len(df0)})")
        print(f"Saved class1 JSON: {json1_path} (objects: {len(class1_records)})")

        return merged_img

if __name__ == "__main__":
    predictor = SlidingWindowPredictor(
        model_path="/home/Mos/Documents/Complex/MyStudy/new_yolo/train_record/poseg/poSeg-original_box_30/weights/best.pt",
        window_size=2560,
        overlap=0,
        device='cpu',  # or 'cuda:0'
        save_window_vis=True
    )
    predictor.predict_and_save(
        image_path="/home/Mos/Desktop/mtemp/complete_test/Experimental_plot_01.png",
        output_dir="/home/Mos/Desktop/mtemp/complete_test/pred5"
    )
