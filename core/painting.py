import cv2
import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.feature import hog
from typing import List
import os

    
class BrushStroke:
    def __init__(self, brush_id, x, y, size, rotation, color, opacity):
        self.brush_id = brush_id
        self.x = int(x)
        self.y = int(y)
        self.size = size
        self.rotation = rotation
        self.color = np.clip(np.array(color, dtype=np.uint8), 0, 255)
        self.opacity = opacity

    def render(self, canvas, brush_library):

        brush_id = min(max(self.brush_id, 0), len(brush_library) - 1)


        if brush_id < 0 or brush_id >= len(brush_library):
            print(f"[⚠️] Invalid brush_id: {brush_id}, skipping stroke.")
            return canvas
            
        brush = brush_library[brush_id]

        h, w = brush.shape[:2]

        # Scale brush
        new_w, new_h = int(w * self.size), int(h * self.size)
         
        if new_w < 1 or new_h < 1:
            return canvas  # skip too small strokes


        canvas_h, canvas_w = canvas.shape[:2]
        x1 = max(0, self.x - new_w // 2)
        y1 = max(0, self.y - new_h // 2)
        x2 = min(canvas_w, self.x + new_w // 2)
        y2 = min(canvas_h, self.y + new_h // 2)

        if x1 >= x2 or y1 >= y2:
            return canvas

        brush_color = np.zeros((new_h, new_w, 3), dtype=float)
        alpha_norm = np.zeros((new_h, new_w, 1), dtype=float)

        if brush.shape[-1] == 4:
            # RGBA brush
            rgb = brush[:, :, :3].astype(float)
            alpha = brush[:, :, 3].astype(float) / 255.0

            rgb_resized = cv2.resize(rgb, (new_w, new_h))
            alpha_resized = cv2.resize(alpha, (new_w, new_h))

            rot_matrix = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), self.rotation, 1.0)
            rgb_rotated = cv2.warpAffine(rgb_resized, rot_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
            alpha_rotated = cv2.warpAffine(alpha_resized, rot_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)

            brush_color = rgb_rotated * (np.array(self.color) / 255.0) * self.opacity
            alpha_norm = (alpha_rotated * self.opacity)[:, :, np.newaxis]

        else:
            # Grayscale brush (fallback)
            gray = cv2.cvtColor(brush, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
            gray_resized = cv2.resize(gray, (new_w, new_h))
            rot_matrix = cv2.getRotationMatrix2D((new_w // 2, new_h // 2), self.rotation, 1.0)
            mask = cv2.warpAffine(gray_resized, rot_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)

            brush_color = np.ones((new_h, new_w, 3), dtype=float) * np.array(self.color) * mask[:, :, np.newaxis] / 255.0
            alpha_norm = (mask * self.opacity)[:, :, np.newaxis]

        region = canvas[y1:y2, x1:x2].astype(float)
        bh, bw = region.shape[:2]
        brush_color = brush_color[:bh, :bw]
        alpha_norm = alpha_norm[:bh, :bw]

        blended = brush_color * alpha_norm + region * (1 - alpha_norm)
        canvas[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        return canvas




def brushes_to_image(brushes: List[BrushStroke], brush_library, canvas_size=(128, 128)):
    canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255
    for stroke in brushes:
        canvas = stroke.render(canvas, brush_library)
    return canvas


def extract_features(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return hog(image_gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)



def load_brush_library(path='brushes/watercolor'):
    files = sorted([f for f in os.listdir(path) if f.endswith('.png')])
    print(f"🖌️ Loaded {len(files)} brushes from {path}")
    brushes = [cv2.imread(os.path.join(path, f), cv2.IMREAD_UNCHANGED) for f in files]
    
    # Optional: warn if any failed to load
    for i, brush in enumerate(brushes):
        if brush is None:
            print(f"⚠️ Warning: Failed to load brush {files[i]}")

    return brushes


def convert_rgb_to_lab(image_rgb):
    return rgb2lab(image_rgb / 255.0)


def convert_lab_to_rgb(image_lab):
    from skimage.color import lab2rgb
    return (lab2rgb(image_lab) * 255).astype(np.uint8)

def compute_edge_map(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return magnitude


def images_to_gif(images, out_path='out.gif', duration=200):
    imgs = [Image.fromarray(img) for img in images]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)



# --- Utility Functions ---
def flatten_genome(brushes):
    genome = []
    for b in brushes:
        genome.extend([
            b.brush_id,
            b.x, b.y,
            b.size,
            b.rotation,
            *b.color,
            b.opacity
        ])
    return genome

def unflatten_genome(genome):
    brushes = []
    for i in range(0, len(genome), 9):
        brushes.append(BrushStroke(
            brush_id=int(genome[i]),
            x=int(genome[i+1]),
            y=int(genome[i+2]),
            size=genome[i+3],
            rotation=genome[i+4],
            color=(int(genome[i+5]), int(genome[i+6]), int(genome[i+7])),
            opacity=genome[i+8]
        ))
    return brushes


# --- Gradient Orientation Map ---
def compute_gradient_orientation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.arctan2(grad_y, grad_x) * 180 / np.pi
    return angles  # shape: (H, W)

