import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .unet import UNet

class AquaEyeBrain:
    def __init__(self, weights_path, device='cpu'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[AquaEye Research Core] Initializing on {self.device}...")
        
        # Initialize U-Net Architecture
        self.model = UNet(n_channels=3, n_classes=1) 
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print("[AquaEye] Neural Weights Loaded. System Ready.")
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to load weights: {e}")
            raise e

    def validate_specimen(self, image_bgr):
        """
        ISO 13322-1 Pre-Check: Ensures image quality meets metrology standards.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 1. Focus Metric (Brenner Gradient for sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 10.0: 
             print(f"[WARNING] Low Focus Score ({laplacian_var:.1f}). Results may be compromised.")
        
        return True

    def preprocess(self, image_bgr):
        self.original_h, self.original_w = image_bgr.shape[:2]
        transform = A.Compose([
            A.Resize(640, 640),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ])
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        augmented = transform(image=image_rgb)
        return augmented['image'].unsqueeze(0).to(self.device)

    def _predict_single_pass(self, image_bgr):
        input_tensor = self.preprocess(image_bgr)
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.sigmoid(output)
        
        probs_np = probs.squeeze().cpu().numpy()
        return cv2.resize(probs_np, (self.original_w, self.original_h), interpolation=cv2.INTER_LINEAR)

    def predict_bayesian_approx(self, image_bgr):
        """
        Monte Carlo Drop-Out Approximation (via TTA)
        Returns: 
          1. Prediction (Mean)
          2. Epistemic Uncertainty (Variance)
        """
        # Rotational TTA (0, 90, 180, 270)
        batch_images = [
            image_bgr,
            cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(image_bgr, cv2.ROTATE_180),
            cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]
        
        preds = []
        # Inference Loop
        for img in batch_images:
            preds.append(self._predict_single_pass(img))
            
        # Re-orient predictions to match original
        aligned_preds = np.array([
            preds[0],
            cv2.rotate(preds[1], cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.rotate(preds[2], cv2.ROTATE_180),
            cv2.rotate(preds[3], cv2.ROTATE_90_CLOCKWISE)
        ])
        
        # Bayesian Statistics
        mean_prediction = np.mean(aligned_preds, axis=0) # The consensus
        uncertainty_map = np.var(aligned_preds, axis=0)  # The confusion
        
        # High-Confidence Mask (Filter out regions where model disagrees with itself)
        # Threshold: Probability > 0.5 AND Variance < 0.1
        final_mask = ((mean_prediction > 0.5) & (uncertainty_map < 0.1)).astype(np.uint8) * 255
        
        return final_mask, mean_prediction, uncertainty_map

    def calculate_morphometrics(self, cnt):
        """
        The Research Engine: Calculates shape descriptors to classify particles.
        References: Zheng et al. (2020), Gerritse et al. (2020)
        """
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        if perimeter == 0: return None
        
        # 1. Circularity (4*pi*A / P^2) - Bubbles are ~1.0
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        # 2. Solidity (Area / ConvexHullArea) - Rough plastics have lower solidity
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # 3. Aspect Ratio (Major / Minor Axis)
        rect = cv2.minAreaRect(cnt)
        (x, y), (w, h), angle = rect
        major = max(w, h)
        minor = min(w, h)
        aspect_ratio = major / minor if minor > 0 else 0
        
        return {
            "circularity": circularity,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "major_axis": major
        }

    def apply_virtual_stain(self, image_bgr):
        """ 🧪 DIGITAL CHEMISTRY: Synthetic Contrast Enhancement """
        # 1. Invert to negatives (plastics often glow in darkfield)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        inverted = cv2.bitwise_not(gray)
        
        # 2. CLAHE (Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(inverted)
        
        # 3. False Color Mapping (Inferno is standard for intensity)
        stained = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
        return stained

    def analyze(self, image_bgr):
        self.validate_specimen(image_bgr)

        # 1. Bayesian Inference (TTA + Uncertainty)
        binary_mask, mean_prob, uncertainty_map = self.predict_bayesian_approx(image_bgr)
        
        # 2. Generate xAI Heatmap (With Transparency for Overlay)
        heatmap_norm = (mean_prob * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        heatmap_bgra = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2BGRA)
        heatmap_bgra[:, :, 3] = (mean_prob * 200).astype(np.uint8) # Transparency

        # 3. Generate Virtual Stain
        virtual_stain = self.apply_virtual_stain(image_bgr)

        # 4. Morphometric Extraction
        # Clean the mask first to remove single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- DATA ACCUMULATORS (The ISO Recorder) ---
        stats = {
            "total_detected": len(contours),
            "rejected_bubbles": 0,
            "rejected_biofilm": 0,
            "accepted_count": 0,
            "morphology": {"fragment": 0, "fiber": 0},
            "particles": [] # Stores raw metrics for the Report Engine
        }
        
        # Output Visualization Canvas
        overlay = np.zeros((self.original_h, self.original_w, 4), dtype=np.uint8)

        for cnt in contours:
            metrics = self.calculate_morphometrics(cnt)
            if not metrics: continue
            
            # --- THE REJECTION LOGIC (Audit Trail) ---
            # We record *why* it was rejected
            if metrics["circularity"] > 0.88:
                stats["rejected_bubbles"] += 1
                continue
            if metrics["solidity"] < 0.5:
                stats["rejected_biofilm"] += 1
                continue
            
            # --- ACCEPTED PARTICLE ---
            stats["accepted_count"] += 1
            stats["particles"].append(metrics) # Save metric data for the PDF
            
            # Classification Logic
            if metrics["aspect_ratio"] > 3.0:
                stats["morphology"]["fiber"] += 1
                color = (0, 255, 255, 255) # Yellow (Fiber)
            else:
                stats["morphology"]["fragment"] += 1
                color = (0, 0, 255, 255)   # Red (Fragment)
            
            # Visualization: Semi-Transparent Fill + Solid Edge
            shape_layer = np.zeros_like(overlay)
            cv2.drawContours(shape_layer, [cnt], -1, color, -1) # Fill
            overlay = cv2.addWeighted(overlay, 1.0, shape_layer, 0.3, 0) # Blend
            cv2.drawContours(overlay, [cnt], -1, color, 2, cv2.LINE_AA) # Edge

        # RETURN: Overlay, Heatmap, Stain, and the DETAILED STATS object
        return overlay, heatmap_bgra, virtual_stain, stats