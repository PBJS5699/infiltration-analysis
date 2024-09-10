from tkinter import Tk, Frame, Button, Radiobutton, Label, Canvas, IntVar, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

root = Tk()
canvas = Canvas(root)
boundary_select = IntVar()
boundary_select.set(1)

# Global variables
image = None
original_image = None
segmenting_color = None
drawing = False
points = []
top_boundary = []
bottom_boundary = []

def create_gui():
    global canvas, boundary_select, color_info, color_display, segmenting_color_label, segmenting_color_display

    button_frame = Frame(root)
    button_frame.pack(side="top", fill="both")

    top_button = Radiobutton(button_frame, text="Top boundary", variable=boundary_select, value=1)
    top_button.pack(side="left")
    bottom_button = Radiobutton(button_frame, text="Bottom boundary", variable=boundary_select, value=2)
    bottom_button.pack(side="left")
    confirm_button = Button(button_frame, text="Confirm", command=calculate_distances_on_segmented_image)
    confirm_button.pack(side="right")
    reset_button = Button(button_frame, text="Reset", command=reset)
    reset_button.pack(side="right")
    segment_button = Button(button_frame, text="Segment", command=segment_by_color_gui)
    segment_button.pack(side="right")

    color_info = Label(button_frame, text="RGB: (---, ---, ---)")
    color_info.pack(side="left")

    color_display = Label(button_frame, width=3, bg="white")
    color_display.pack(side="left")

    segmenting_color_label = Label(button_frame, text="Segmenting Color: None")
    segmenting_color_label.pack(side="left")

    segmenting_color_display = Label(button_frame, width=3, bg="white")
    segmenting_color_display.pack(side="left")

    canvas.pack(side="top")

    canvas.bind("<ButtonPress-1>", start_draw)
    canvas.bind("<ButtonRelease-1>", stop_draw)
    canvas.bind("<B1-Motion>", draw_line)
    canvas.bind("<ButtonPress-3>", save_segmenting_color)
    canvas.bind("<Motion>", update_color_display)

def update_image(image, canvas, temp_points, color):
    temp_image = image.copy()
    for i in range(len(temp_points) - 1):
        cv2.line(temp_image, temp_points[i], temp_points[i + 1], color, 2)
    cv2image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, image=imgtk, anchor="nw")
    return imgtk

def update_color_display(event):
    global image, color_info, color_display
    x, y = event.x, event.y
    if image is not None and x < image.shape[1] and y < image.shape[0]:
        b, g, r = image[y, x]
        rgb_str = f"RGB: ({r}, {g}, {b})"
        color_info.config(text=rgb_str)
        color_display.config(bg=f"#{r:02x}{g:02x}{b:02x}")
    else:
        color_info.config(text="RGB: (---, ---, ---)")
        color_display.config(bg="white")

def update_segmenting_color_display():
    global segmenting_color, segmenting_color_display, segmenting_color_label
    if segmenting_color:
        color_hex = f"#{segmenting_color[0]:02x}{segmenting_color[1]:02x}{segmenting_color[2]:02x}"
        segmenting_color_display.config(bg=color_hex)
        segmenting_color_label.config(text=f"Segmenting Color: {segmenting_color}")
    else:
        segmenting_color_display.config(bg="white")
        segmenting_color_label.config(text="Segmenting Color: None")

def start_draw(event):
    global drawing, points, top_boundary, bottom_boundary
    points = []
    if boundary_select.get() == 1:
        top_boundary = []
    elif boundary_select.get() == 2:
        bottom_boundary = []
    drawing = True
    draw_line(event)

def stop_draw(event):
    global drawing, top_boundary, bottom_boundary
    drawing = False
    draw_line(event)
    if boundary_select.get() == 1:
        top_boundary = points.copy()
        print("Top boundary updated:", top_boundary)
    elif boundary_select.get() == 2:
        bottom_boundary = points.copy()
        print("Bottom boundary updated:", bottom_boundary)

def draw_line(event):
    global drawing, points
    x, y = event.x, event.y
    if drawing:
        points.append((x, y))

def save_segmenting_color(event):
    global segmenting_color, image
    x, y = event.x, event.y
    if image is not None and x < image.shape[1] and y < image.shape[0]:
        b, g, r = image[y, x]
        segmenting_color = (r, g, b)
        update_segmenting_color_display()
    else:
        segmenting_color = None
        update_segmenting_color_display()

def reset():
    global image, original_image, top_boundary, bottom_boundary, segmenting_color
    if original_image is not None:
        image = original_image.copy()
    segmenting_color = None
    top_boundary = []
    bottom_boundary = []
    update_image(image, canvas, [], (0, 0, 0))
    update_segmenting_color_display()

def calculate_distances_on_segmented_image():
    global image, segmenting_color, top_boundary, bottom_boundary
    from utils import calculate_distance  # Import here to avoid circular imports
    
    if segmenting_color is None or (original_image is not None and np.array_equal(image, original_image)):
        messagebox.showinfo("Segmentation Required", "Please select a segmenting color and segment the image first")
        return
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    calculate_distance(gray_image, top_boundary, bottom_boundary)

def segment_by_color_gui():
    global image, original_image, segmenting_color
    from utils import segment_by_color  # Import here to avoid circular imports
    
    if original_image is not None and segmenting_color is not None:
        image = segment_by_color(image, original_image, segmenting_color)
        update_image(image, canvas, [], (0, 0, 0))

# Export necessary variables and functions
__all__ = ['root', 'canvas', 'boundary_select', 'create_gui', 'update_image', 'update_color_display', 
           'update_segmenting_color_display', 'start_draw', 'stop_draw', 'draw_line', 
           'save_segmenting_color', 'reset', 'calculate_distances_on_segmented_image', 'segment_by_color_gui',
           'image', 'original_image', 'segmenting_color', 'drawing', 'points', 'top_boundary', 'bottom_boundary']