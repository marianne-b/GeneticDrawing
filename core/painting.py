import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import matplotlib.animation as animation
from abc import ABC, abstractmethod
from geneticdrawing import Brush


class Painting(ABC):
    """
    Abstract base class for paintings.
    A painting is a collection of brushes that can be rendered.
    """
    
    def __init__(self, canvas_size=(800, 800), background_color=(255, 255, 255, 255)):
        """
        Initialize a painting.
        
        Args:
            canvas_size (tuple): Size of the canvas as (width, height)
            background_color (tuple): RGBA background color
        """
        self.canvas_size = canvas_size
        self.background_color = background_color
        self.brushes = []  # List to store brushes in order of drawing
        self.brush_ids = {}  # Dictionary to map IDs to brush indices
        self.next_id = 1  # For generating unique IDs for brushes
        
        # Create the initial canvas
        self.canvas = Image.new('RGBA', canvas_size, background_color)
    
    def add_brush(self, brush, color=(0, 0, 0, 255), steps=100):
        """
        Add a brush to the painting.
        
        Args:
            brush: Brush object to add
            color (tuple): RGBA color for the brush
            steps (int): Number of steps for rendering the brush
            
        Returns:
            int: Unique ID for the added brush
        """
        # Make a copy of the brush to avoid modifying the original
        brush_copy = brush.copy()
        
        # Generate a unique ID for this brush
        brush_id = self.next_id
        self.next_id += 1
        
        # Calculate brush size for ordering (approximated as max distance between points)
        size = self._calculate_brush_size(brush_copy)
        
        # Store the brush info
        brush_info = {
            'brush': brush_copy,
            'color': color,
            'steps': steps,
            'id': brush_id,
            'size': size
        }
        
        self.brushes.append(brush_info)
        self.brush_ids[brush_id] = len(self.brushes) - 1
        
        return brush_id
    
    def _calculate_brush_size(self, brush):
        """Calculate the approximate size of a brush based on its points"""
        if not brush.points or len(brush.points) < 2:
            return 0
            
        # Find the maximum distance between any two points
        max_distance = 0
        for i in range(len(brush.points)):
            for j in range(i+1, len(brush.points)):
                p1 = brush.points[i]["pos"]
                p2 = brush.points[j]["pos"]
                dist = np.linalg.norm(p2 - p1)
                max_distance = max(max_distance, dist)
                
        return max_distance
    
    def get_brush(self, brush_id):
        """
        Get a brush by its ID.
        
        Args:
            brush_id (int): ID of the brush
            
        Returns:
            brush: The brush object for direct modification
        """
        if brush_id in self.brush_ids:
            index = self.brush_ids[brush_id]
            return self.brushes[index]['brush']
        else:
            raise ValueError(f"No brush found with ID {brush_id}")
    
    def update_brush_properties(self, brush_id, color=None, steps=None):
        """
        Update properties of a brush.
        
        Args:
            brush_id (int): ID of the brush to update
            color (tuple, optional): New RGBA color
            steps (int, optional): New step count for rendering
            
        Returns:
            self: For method chaining
        """
        if brush_id in self.brush_ids:
            index = self.brush_ids[brush_id]
            if color is not None:
                self.brushes[index]['color'] = color
            if steps is not None:
                self.brushes[index]['steps'] = steps
                
            # Update the brush size after modifications
            self.brushes[index]['size'] = self._calculate_brush_size(self.brushes[index]['brush'])
        
        return self
    
    def remove_brush(self, brush_id):
        """
        Remove a brush from the painting.
        
        Args:
            brush_id (int): ID of the brush to remove
            
        Returns:
            self: For method chaining
        """
        if brush_id in self.brush_ids:
            index = self.brush_ids[brush_id]
            self.brushes.pop(index)
            
            # Update the brush_ids dictionary
            self.brush_ids = {}
            for i, brush_info in enumerate(self.brushes):
                self.brush_ids[brush_info['id']] = i
        
        return self
    
    def clear(self):
        """
        Clear the canvas and remove all brushes.
        
        Returns:
            self: For method chaining
        """
        self.brushes = []
        self.brush_ids = {}
        self.next_id = 1
        self.canvas = Image.new('RGBA', self.canvas_size, self.background_color)
        return self
    
    def set_background(self, color):
        """
        Set the background color of the canvas.
        
        Args:
            color (tuple): RGBA background color
            
        Returns:
            self: For method chaining
        """
        self.background_color = color
        return self
    
    def render(self):
        """
        Render all brushes to the canvas in order.
        
        Returns:
            PIL.Image: The rendered painting
        """
        # Start with a fresh canvas
        canvas = Image.new('RGBA', self.canvas_size, self.background_color)
        
        # Render each brush onto the canvas in order
        for brush_info in self.brushes:
            brush = brush_info['brush']
            color = brush_info['color']
            steps = brush_info['steps']
            
            # Render the brush onto its own image
            brush_image = brush._build(canvas_size=self.canvas_size, brush_color=color, steps=steps)
            
            # Composite the brush onto the canvas
            canvas = Image.alpha_composite(canvas, brush_image)
        
        # Store the rendered canvas
        self.canvas = canvas
        
        return canvas
    
    def display(self):
        """
        Render and display the painting.
        
        Returns:
            self: For method chaining
        """
        # Render the painting
        self.render()
        
        # Display with matplotlib
        plt.figure(figsize=(10, 10))
        plt.imshow(self.canvas)
        plt.axis('off')
        plt.show()
        
        return self
    
    def display_process(self, duration=5000, fps=10):
        """
        Display an animation of the painting process from largest to smallest brush.
        
        Args:
            duration (int): Duration of the animation in milliseconds
            fps (int): Frames per second
            
        Returns:
            self: For method chaining
        """
        if not self.brushes:
            print("No brushes to animate")
            return self
            
        # Sort brushes by size (from largest to smallest)
        sorted_brushes = sorted(self.brushes, key=lambda b: b['size'], reverse=True)
        
        # Create figures for animation
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis('off')
        
        # Function to create each frame
        def update(frame):
            # Clear the axis
            ax.clear()
            ax.axis('off')
            
            # Start with a fresh canvas
            canvas = Image.new('RGBA', self.canvas_size, self.background_color)
            
            # Add brushes up to the current frame
            for i in range(min(frame + 1, len(sorted_brushes))):
                brush_info = sorted_brushes[i]
                brush = brush_info['brush']
                color = brush_info['color']
                steps = brush_info['steps']
                
                # Render the brush
                brush_image = brush._build(canvas_size=self.canvas_size, brush_color=color, steps=steps)
                
                # Composite the brush onto the canvas
                canvas = Image.alpha_composite(canvas, brush_image)
            
            # Display the current state
            ax.imshow(canvas)
            return [ax]
        
        # Create the animation
        frames = len(sorted_brushes)
        interval = duration / frames
        anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
        
        plt.show()
        
        return self
    
    def save(self, filename):
        """
        Render and save the painting to a file.
        
        Args:
            filename (str): Output filename
            
        Returns:
            self: For method chaining
        """
        # Render the painting
        self.render()
        
        # Save the image
        self.canvas.save(filename)
        print(f"Painting saved to {filename}")
        
        return self
    
    def save_process_gif(self, filename, duration=5000):
        """
        Save an animation of the painting process as a GIF.
        
        Args:
            filename (str): Output filename
            duration (int): Duration of the animation in milliseconds
            
        Returns:
            self: For method chaining
        """
        if not self.brushes:
            print("No brushes to animate")
            return self
            
        # Sort brushes by size (from largest to smallest)
        sorted_brushes = sorted(self.brushes, key=lambda b: b['size'], reverse=True)
        
        # Create frames for the GIF
        frames = []
        
        for i in range(len(sorted_brushes)):
            # Start with a fresh canvas
            canvas = Image.new('RGBA', self.canvas_size, self.background_color)
            
            # Add brushes up to the current frame
            for j in range(i + 1):
                brush_info = sorted_brushes[j]
                brush = brush_info['brush']
                color = brush_info['color']
                steps = brush_info['steps']
                
                # Render the brush
                brush_image = brush._build(canvas_size=self.canvas_size, brush_color=color, steps=steps)
                
                # Composite the brush onto the canvas
                canvas = Image.alpha_composite(canvas, brush_image)
            
            frames.append(canvas)
        
        # Calculate frame duration (ms)
        frame_duration = duration // len(frames)
        
        # Save as GIF
        frames[0].save(
            filename,
            save_all=True,
            append_images=frames[1:],
            optimize=False,
            duration=frame_duration,
            loop=0
        )
        
        print(f"Animation saved to {filename}")
        return self
    
    @abstractmethod
    def flatten(self):
        """
        Create a flattened representation of the painting.
        
        Returns:
            Any: Flattened representation (e.g., a vector of parameters)
        """
        pass
    
    @abstractmethod
    def unflatten(self, flattened_data):
        """
        Update the painting from a flattened representation.
        
        Args:
            flattened_data: The flattened representation
            
        Returns:
            self: For method chaining
        """
        pass
    
    def get_brush_info(self, brush_id):
        """
        Get information about a brush by its ID.
        
        Args:
            brush_id (int): ID of the brush
            
        Returns:
            dict: Brush information dictionary
        """
        if brush_id in self.brush_ids:
            index = self.brush_ids[brush_id]
            return self.brushes[index]
        return None
    
    def get_all_brushes(self):
        """
        Get all brushes in the painting.
        
        Returns:
            list: List of brush information dictionaries
        """
        return self.brushes
    
    def resize(self, new_size):
        """
        Resize the canvas.
        
        Args:
            new_size (tuple): New size as (width, height)
            
        Returns:
            self: For method chaining
        """
        self.canvas_size = new_size
        return self

    def __add__(self, other):
        """
        Add two paintings together by combining their brushes.
        
        Args:
            other (Painting): The painting to add to this one
            
        Returns:
            Painting: A new painting containing brushes from both paintings
        """
        if not isinstance(other, Painting):
            raise TypeError(f"Cannot add Painting and {type(other)}")
        
        # Create a new painting with the same properties as this one
        # Note: This will need to be adapted for concrete subclasses
        if type(self) is type(other):
            # If both paintings are of the same concrete type, use that type
            result = type(self)(canvas_size=self.canvas_size, background_color=self.background_color)
            result.clear()  # Clear any default brushes that might have been created in the constructor
        else:
            # If types differ, create a basic Painting - may need custom handling for specific subclasses
            from copy import deepcopy
            # We can't instantiate the abstract Painting class directly, so we'll have to
            # create a concrete instance and then clear it
            result = deepcopy(self)
            result.clear()
        
        # Add brushes from this painting
        for brush_info in self.brushes:
            brush_copy = brush_info['brush'].copy()
            result.add_brush(
                brush_copy, 
                color=brush_info['color'],
                steps=brush_info['steps']
            )
        
        # Add brushes from the other painting
        for brush_info in other.brushes:
            brush_copy = brush_info['brush'].copy()
            result.add_brush(
                brush_copy, 
                color=brush_info['color'],
                steps=brush_info['steps']
            )
        
        return result

    def __iadd__(self, other):
        """
        In-place addition of another painting by adding its brushes to this one.
        
        Args:
            other (Painting): The painting whose brushes to add to this one
            
        Returns:
            self: This painting with added brushes
        """
        if not isinstance(other, Painting):
            raise TypeError(f"Cannot add Painting and {type(other)}")
        
        # Add brushes from the other painting to this one
        for brush_info in other.brushes:
            brush_copy = brush_info['brush'].copy()
            self.add_brush(
                brush_copy, 
                color=brush_info['color'],
                steps=brush_info['steps']
            )
        
        return self
        
    def apply(self, transformation_fn):
        """
        Apply a transformation function to all brushes in the painting.
        
        Args:
            transformation_fn: A function that takes a brush object and applies transformations to it.
                              The function should modify the brush in-place and doesn't need to return anything.
        
        Returns:
            self: The painting with transformed brushes, for method chaining
        """
        # Apply the transformation to each brush
        for brush_info in self.brushes:
            brush = brush_info['brush']
            transformation_fn(brush)
            
            # Update the brush size after transformation
            brush_info['size'] = self._calculate_brush_size(brush)
        
        return self


class SquarePainting(Painting):
    """
    A concrete implementation of a painting that represents a square.
    """
    
    def __init__(self, canvas_size=(800, 800), background_color=(255, 255, 255, 255), 
                 size=0.8, color=(0, 0, 0, 255)):
        """
        Initialize a square painting.
        
        Args:
            canvas_size (tuple): Size of the canvas
            background_color (tuple): RGBA background color
            size (float): Size of the square (from -1 to 1)
            color (tuple): RGBA color for the square
        """
        super().__init__(canvas_size, background_color)
        
        # Create the square with the given size
        self.create_square(size, color)
    
    def create_square(self, size, color=(0, 0, 0, 255)):
        """
        Create a square with the given size and color.
        
        Args:
            size (float): Size of the square (from -1 to 1)
            color (tuple): RGBA color for the square
            
        Returns:
            self: For method chaining
        """
        # Clear any existing brushes
        self.clear()
        
        # Create the square brushes
        self._create_edge("top", size, color)
        self._create_edge("right", size, color)
        self._create_edge("bottom", size, color)
        self._create_edge("left", size, color)
        
        return self
    
    def _create_edge(self, edge_type, size, color=(0, 0, 0, 255)):
        """
        Create a brush for a square edge.
        
        Args:
            edge_type (str): Type of edge ("top", "right", "bottom", "left")
            size (float): Size of the square
            color (tuple): RGBA color for the edge
            
        Returns:
            int: ID of the created brush
        """
        
        # Define the corners of the square
        half_size = size / 2
        top_left = (-half_size, -half_size)
        top_right = (half_size, -half_size)
        bottom_right = (half_size, half_size)
        bottom_left = (-half_size, half_size)
        
        # Create the appropriate brush based on edge type
        if edge_type == "top":
            brush = Brush(top_left, top_right, scale=1.0, opacity=1.0)
        elif edge_type == "right":
            brush = Brush(top_right, bottom_right, scale=1.0, opacity=1.0)
        elif edge_type == "bottom":
            brush = Brush(bottom_right, bottom_left, scale=1.0, opacity=1.0)
        elif edge_type == "left":
            brush = Brush(bottom_left, top_left, scale=1.0, opacity=1.0)
        else:
            raise ValueError(f"Unknown edge type: {edge_type}")
        
        # Add the brush to the painting
        return self.add_brush(brush, color=color)
    
    def set_edge_opacity(self, edge_type, opacity):
        """
        Set the opacity of a specific edge.
        
        Args:
            edge_type (str): Type of edge ("top", "right", "bottom", "left")
            opacity (float): Opacity value (0.0 to 1.0)
            
        Returns:
            self: For method chaining
        """
        # Map edge types to brush indices (assuming the order they were created)
        edge_indices = {"top": 0, "right": 1, "bottom": 2, "left": 3}
        
        if edge_type not in edge_indices:
            raise ValueError(f"Unknown edge type: {edge_type}")
        
        # Get the brush ID from its index
        for brush_info in self.brushes:
            if self.brush_ids[brush_info['id']] == edge_indices[edge_type]:
                brush_id = brush_info['id']
                brush = self.get_brush(brush_id)
                brush.set_opacity(opacity)
                return self
        
        return self
    
    def resize_square(self, new_size):
        """
        Resize the square to a new size.
        
        Args:
            new_size (float): New size for the square
            
        Returns:
            self: For method chaining
        """
        # Get the current size (from the first brush)
        if not self.brushes:
            return self
            
        current_size = self.brushes[0]['size']
        scale_factor = new_size / current_size if current_size != 0 else 1.0
        
        # Scale all brushes
        for brush_info in self.brushes:
            brush = self.get_brush(brush_info['id'])
            brush.scale_brush(scale_factor)
            
            # Update the size
            brush_info['size'] = self._calculate_brush_size(brush)
        
        return self
    
    def flatten(self):
        """
        Create a flattened representation of the square painting.
        The flattened vector is [size, top_opacity, right_opacity, bottom_opacity, left_opacity].
        
        Returns:
            list: Flattened representation
        """
        if len(self.brushes) != 4:
            raise ValueError("Square painting should have exactly 4 brushes (edges)")
        
        # Get the brushes (assuming they are ordered: top, right, bottom, left)
        brushes = self.get_all_brushes()
        
        # Extract size from the first brush (assuming all sides have the same size)
        size = brushes[0]['size']
        
        # Extract opacity from each brush
        opacities = [brush['brush'].opacity for brush in brushes]
        
        # Return the flattened representation
        return [size] + opacities
    
    def unflatten(self, flattened_data):
        """
        Update the square painting from a flattened representation.
        
        Args:
            flattened_data (list): Flattened representation [size, top_opacity, right_opacity, bottom_opacity, left_opacity]
            
        Returns:
            self: For method chaining
        """
        if len(self.brushes) != 4:
            raise ValueError("Square painting should have exactly 4 brushes (edges)")
        
        # Unpack the flattened data
        size, top_opacity, right_opacity, bottom_opacity, left_opacity = flattened_data
        
        # Resize the square
        self.resize_square(size)
        
        # Update opacities
        self.set_edge_opacity("top", top_opacity)
        self.set_edge_opacity("right", right_opacity)
        self.set_edge_opacity("bottom", bottom_opacity)
        self.set_edge_opacity("left", left_opacity)
        
        return self
