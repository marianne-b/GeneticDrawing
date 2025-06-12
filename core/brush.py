import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image, ImageDraw
from math import cos, sin, radians, tan
import copy

class Brush:
    def __init__(self, p0, p1, scale=1.0, opacity=1.0):
        self.points = [
            {"pos": np.array(p0), "width": 1.0},
            {"pos": np.array(p1), "width": 1.0}
        ]
        
        self.scale_brush(scale)
        
        self.custom_weights = {}
        self.curvatures = {}
        self.opacity = opacity

    def add_point(self, point, width=1.0):
        self.points.append({"pos": np.array(point), "width": width})
        return self
    
    def set_width(self, index, width):
        if 0 <= index < len(self.points):
            self.points[index]["width"] = width
        else:
            raise IndexError(f"Point index {index} out of range")
        return self
    
    def set_weight(self, i, j, weight):
        if 0 <= i < len(self.points) and 0 <= j < len(self.points):
            self.custom_weights[(i, j)] = weight
            self.custom_weights[(j, i)] = weight
        else:
            raise IndexError(f"Point indices {i}, {j} out of range")
        return self
    
    def set_curvature(self, i, j, curvature):
        curvature = max(-1.0, min(1.0, curvature))
        
        if 0 <= i < len(self.points) and 0 <= j < len(self.points):
            self.curvatures[(i, j)] = curvature
            self.curvatures[(j, i)] = curvature
        else:
            raise IndexError(f"Point indices {i}, {j} out of range")
        return self
    
    def set_opacity(self, opacity):
        self.opacity = max(0.0, min(1.0, opacity))
        return self
    
    def mat_transform(self, matrix, center=(0, 0)):
        center_np = np.array(center)
        
        for point in self.points:
            translated = point["pos"] - center_np
            transformed = matrix @ np.append(translated, 1)
            point["pos"] = transformed[:2] + center_np
            
        return self
    
    def rotate(self, angle_deg, center=(0, 0)):
        angle_rad = radians(angle_deg)
        
        rotation_matrix = np.array([
            [cos(angle_rad), -sin(angle_rad), 0],
            [sin(angle_rad), cos(angle_rad), 0],
            [0, 0, 1]
        ])
        
        return self.mat_transform(rotation_matrix, center)
    
    def scale_brush(self, factor):
        if isinstance(factor, (int, float)):
            factor_x = factor_y = factor
        else:
            factor_x, factor_y = factor
            
        scale_matrix = np.array([
            [factor_x, 0, 0],
            [0, factor_y, 0],
            [0, 0, 1]
        ])
        
        return self.mat_transform(scale_matrix)
    
    def translate(self, dx, dy):
        translation_matrix = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ])
        
        return self.mat_transform(translation_matrix)
    
    def shear(self, shear_x=0, shear_y=0):
        shear_matrix = np.array([
            [1, shear_x, 0],
            [shear_y, 1, 0],
            [0, 0, 1]
        ])
        
        return self.mat_transform(shear_matrix)
    
    def reflect_x(self):
        reflection_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        
        return self.mat_transform(reflection_matrix)
    
    def reflect_y(self):
        reflection_matrix = np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        return self.mat_transform(reflection_matrix)
    
    def reflect_origin(self):
        reflection_matrix = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
        
        return self.mat_transform(reflection_matrix)
    
    def reflect_line(self, angle_deg):
        angle_rad = radians(angle_deg)
        cos_a = cos(2 * angle_rad)
        sin_a = sin(2 * angle_rad)
        
        reflection_matrix = np.array([
            [cos_a, sin_a, 0],
            [sin_a, -cos_a, 0],
            [0, 0, 1]
        ])
        
        return self.mat_transform(reflection_matrix)
    
    def _compute_mst(self):
        G = nx.Graph()
        
        for i in range(len(self.points)):
            G.add_node(i)
        
        for i in range(len(self.points)):
            for j in range(i+1, len(self.points)):
                if (i, j) in self.custom_weights:
                    weight = self.custom_weights[(i, j)]
                else:
                    p1 = self.points[i]["pos"]
                    p2 = self.points[j]["pos"]
                    weight = np.linalg.norm(p2 - p1)
                
                G.add_edge(i, j, weight=weight)
        
        mst = nx.minimum_spanning_tree(G)
        
        return list(mst.edges())
    
    def _bezier_point(self, p0, p1, control, t):
        return (1-t)**2 * p0 + 2*(1-t)*t * control + t**2 * p1
    
    def _compute_control_point(self, p0, p1, curvature):
        mid = (p0 + p1) / 2
        
        direction = p1 - p0
        perpendicular = np.array([-direction[1], direction[0]])
        
        if np.linalg.norm(perpendicular) > 0:
            perpendicular = perpendicular / np.linalg.norm(perpendicular)
        
        distance = np.linalg.norm(p1 - p0)
        offset = perpendicular * distance * curvature * 0.5
        
        return mid + offset
    
    def _interpolate_width(self, width1, width2, t):
        return width1 * (1-t) + width2 * t
    
    def _build(self, canvas_size=(800, 800), brush_color=(0, 0, 0, 255), steps=100):
        image = Image.new('RGBA', canvas_size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        
        opacity = getattr(self, 'opacity', 1.0)
        
        if isinstance(brush_color, tuple) and len(brush_color) == 4:
            r, g, b, a = brush_color
            a = int(a * opacity)
            brush_color = (r, g, b, a)
        
        mst_edges = self._compute_mst()
        
        def to_canvas(point):
            x = (point[0] + 1) * (canvas_size[0] / 2)
            y = (point[1] + 1) * (canvas_size[1] / 2)
            return (x, y)
        
        for i, j in mst_edges:
            p0 = self.points[i]["pos"]
            p1 = self.points[j]["pos"]
            width0 = self.points[i]["width"]
            width1 = self.points[j]["width"]
            
            curvature = self.curvatures.get((i, j), 0)
            
            control = self._compute_control_point(p0, p1, curvature)
            
            # Generate points along the curve for a smooth path
            curve_points = []
            widths = []
            
            for step in range(steps + 1):
                t = step / steps
                
                point = self._bezier_point(p0, p1, control, t)
                width = self._interpolate_width(width0, width1, t)
                
                curve_points.append(to_canvas(point))
                widths.append(width)
            
            # Create smooth stroke by drawing polygons between consecutive points
            for k in range(len(curve_points) - 1):
                # Get current and next point
                p_curr = np.array(curve_points[k])
                p_next = np.array(curve_points[k+1])
                
                # Calculate direction vector
                direction = p_next - p_curr
                length = np.linalg.norm(direction)
                if length < 1e-6:  # Skip if points are too close
                    continue
                
                # Normalize direction vector
                direction = direction / length
                
                # Get perpendicular vector
                perp = np.array([-direction[1], direction[0]])
                
                # Get widths at current and next point
                w_curr = widths[k] * min(canvas_size) / 50
                w_next = widths[k+1] * min(canvas_size) / 50
                
                # Calculate the four corners of the quad
                p1 = p_curr + perp * w_curr/2
                p2 = p_curr - perp * w_curr/2
                p3 = p_next - perp * w_next/2
                p4 = p_next + perp * w_next/2
                
                # Draw as a polygon
                draw.polygon([tuple(p1), tuple(p4), tuple(p3), tuple(p2)], fill=brush_color)
                
                # Draw a circle at each junction to smooth transitions
                radius = max(w_curr, w_next) / 2
                draw.ellipse(
                    (
                        p_next[0] - radius,
                        p_next[1] - radius,
                        p_next[0] + radius,
                        p_next[1] + radius
                    ),
                    fill=brush_color
                )
            
            # Draw a circle at the start point to cap the stroke
            radius = widths[0] * min(canvas_size) / 50 / 2
            draw.ellipse(
                (
                    curve_points[0][0] - radius,
                    curve_points[0][1] - radius,
                    curve_points[0][0] + radius,
                    curve_points[0][1] + radius
                ),
                fill=brush_color
            )
        
        return image
    
    def save(self, filename='brush.png', canvas_size=(800, 800), brush_color=(0, 0, 0, 255), steps=100):
        image = self._build(canvas_size, brush_color, steps)
        image.save(filename)
        print(f"Brush stroke saved to {filename}")
        return self
    
    def show(self, canvas_size=(800, 800), brush_color=(0, 0, 0, 255), steps=100):
        image = self._build(canvas_size, brush_color, steps)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        return self

    
    def copy(self):
        """
        Create a deep copy of this brush.
        
        Returns:
            Brush: A new brush with the same properties
        """
        # Use deep copy to ensure all nested structures are copied
        return copy.deepcopy(self)
    
    @staticmethod
    def paint(brushes, canvas_size=(800, 800), background_color=(255, 255, 255, 255)):
        canvas = Image.new('RGBA', canvas_size, background_color)
        
        for brush_info in brushes:
            if isinstance(brush_info, tuple) and len(brush_info) >= 2:
                brush = brush_info[0]
                brush_color = brush_info[1]
                steps = brush_info[2] if len(brush_info) > 2 else 100
            else:
                brush = brush_info
                brush_color = (0, 0, 0, 255)
                steps = 100
            
            brush_image = brush._build(canvas_size=canvas_size, brush_color=brush_color, steps=steps)
            
            canvas = Image.alpha_composite(canvas, brush_image)
        
        return canvas
    
    @staticmethod
    def show_painting(brushes, canvas_size=(800, 800), background_color=(255, 255, 255, 255)):
        canvas = Brush.paint(brushes, canvas_size, background_color)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(canvas)
        plt.axis('off')
        plt.show()
        
        return canvas
    
    @staticmethod
    def save_painting(filename, brushes, canvas_size=(800, 800), background_color=(255, 255, 255, 255)):
        canvas = Brush.paint(brushes, canvas_size, background_color)
        canvas.save(filename)
        print(f"Painting saved to {filename}")
        
        return canvas

    
