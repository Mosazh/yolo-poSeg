import os
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
        Run sliding-window inference, save merged visualization and CSV of coords.

        - cls=0: saves bounding-box center points.
        - cls=1: saves each mask pixel coordinate.

        Returns:
            merged_img (np.ndarray): Final blended image array.
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # Records for all coords
        records = []

        # Read image
        img = cv2.imread(image_path)
        full_h, full_w = img.shape[:2]

        # Prepare for blending
        merged_sum = np.zeros_like(img, dtype=float)
        weight_sum = np.zeros((full_h, full_w), dtype=float)

        # Slide over windows
        windows = self._sliding_windows(img.shape)
        for idx, (x1, y1, x2, y2) in enumerate(tqdm(windows, desc="Sliding Windows")):
            crop = img[y1:y2, x1:x2]
            # Inference
            results = self.model(crop, imgsz=self.window_size, conf=0.34, device=self.device, verbose=False)
            res = results[0]

            # Extract masks if available
            masks = None
            if hasattr(res, 'masks') and res.masks is not None:
                # masks.data: (n_masks, win_h, win_w)
                masks = res.masks.data.cpu().numpy().astype(np.uint8)

            # Process each detected object
            for i, box in enumerate(res.boxes):
                cls_idx = int(box.cls.item())
                # Class 0 -> save bounding-box center
                if cls_idx == 0:
                    x1b, y1b, x2b, y2b = box.xyxy[0].tolist()
                    cx = x1 + (x1b + x2b) / 2
                    cy = y1 + (y1b + y2b) / 2
                    records.append({
                        'cls': 0,
                        'x': float(cx),
                        'y': float(cy),
                        'window': f"({x1},{y1})"
                    })
                # Class 1 -> save each mask pixel
                elif cls_idx == 1 and masks is not None:
                    mask_i = masks[i]  # 2D binary mask for this object
                    ys, xs = np.where(mask_i > 0)
                    # Append each mask pixel coordinate
                    for yy, xx in zip(ys, xs):
                        records.append({
                            'cls': 1,
                            'x': int(x1 + xx),
                            'y': int(y1 + yy),
                            'window': f"({x1},{y1})"
                        })

            # Optional: save window-level visualization
            vis = res.plot()
            if self.save_window_vis:
                win_vis_path = os.path.join(output_dir, f"{base_name}_win{idx}.jpg")
                cv2.imwrite(win_vis_path, vis)

            # Blend window visualization into full image
            vis_float = vis.astype(float)
            merged_sum[y1:y2, x1:x2] += vis_float * 0.7
            weight_sum[y1:y2, x1:x2] += 0.7

        # Finalize blended result
        # Avoid division by zero
        weight_sum_expanded = np.expand_dims(weight_sum, axis=2)
        blended = np.divide(merged_sum, weight_sum_expanded,
                            out=np.zeros_like(merged_sum), where=weight_sum_expanded>0)
        merged_img = blended.astype(np.uint8)

        # Save merged image
        merged_path = os.path.join(output_dir, f"{base_name}_merged.jpg")
        cv2.imwrite(merged_path, merged_img)

        # Save coordinates to CSV
        df = pd.DataFrame(records)
        csv_path = os.path.join(output_dir, f"{base_name}_coords.csv")
        df.to_csv(csv_path, index=False)

        print(f"Saved merged image: {merged_path}")
        print(f"Saved coordinates CSV: {csv_path} (total records: {len(df)})")

        return merged_img


if __name__ == "__main__":
    predictor = SlidingWindowPredictor(
        model_path="/home/Mos/Documents/Complex/MyStudy/new_yolo/train_record/poseg/poSeg-original_box_30/weights/best.pt",
        window_size=2560,
        overlap=0,
        device='cpu',  # or 'cuda:0'
        save_window_vis=False
    )
    predictor.predict_and_save(
        image_path="/home/Mos/Desktop/mtemp/complete_test/Experimental_plot_01r.png",
        output_dir="/home/Mos/Desktop/mtemp/complete_test/pred5"
    )
