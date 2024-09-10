import os
from tkinter import *
from tkinter import filedialog, Canvas
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import TclError, colorchooser
from tkinter import Tk
from PIL import ImageTk
from matplotlib.path import Path

# Global variables
root = Tk()
image_boundaries = {}
points = []
top_boundary = []
bottom_boundary = []
boundaries_list = []
drawing = False
segmenting_color = None
boundary_select = IntVar()
boundary_select.set(1)
continue_to_next = []
all_distances_df = pd.DataFrame()

# Functions

def segment_by_color():
    global image
    if segmenting_color is None:
        image = original_image.copy()
    else:
        # Define the range of color to be segmented
        range = 40
        color_lower = np.array(
            [
                max(segmenting_color[2] - range, 0),
                max(segmenting_color[1] - range, 0),
                max(segmenting_color[0] - range, 0),
            ],
            dtype="uint8",
        )
        color_upper = np.array(
            [
                min(segmenting_color[2] + range, 255),
                min(segmenting_color[1] + range, 255),
                min(segmenting_color[0] + range, 255),
            ],
            dtype="uint8",
        )

        # Create a mask that marks the pixels within the range
        mask = cv2.inRange(image, color_lower, color_upper)

        # Apply Otsu's thresholding to the mask
        _, otsu_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Convert the mask to a BGR image so it can be displayed using ImageTk
        segmented_image = cv2.cvtColor(otsu_mask, cv2.COLOR_GRAY2BGR)
        image = segmented_image
    update_image(image, canvas, [], (0, 0, 0))


def save_segmenting_color(event):
    global segmenting_color
    x, y = event.x, event.y
    if x < image.shape[1] and y < image.shape[0]:
        b, g, r = image[y, x]
        segmenting_color = (r, g, b)
        update_segmenting_color_display()
    else:
        segmenting_color = None
        update_segmenting_color_display()


def update_segmenting_color_display():
    if segmenting_color:
        color_hex = f"#{segmenting_color[0]:02x}{segmenting_color[1]:02x}{segmenting_color[2]:02x}"
        segmenting_color_display.config(bg=color_hex)
        segmenting_color_label.config(text=f"Segmenting Color: {segmenting_color}")
    else:
        segmenting_color_display.config(bg="white")
        segmenting_color_label.config(text="Segmenting Color: None")


def update_color_display(event):
    global image
    x, y = event.x, event.y
    if x < image.shape[1] and y < image.shape[0]:
        # Get pixel value at (x, y)
        b, g, r = image[y, x]
        rgb_str = f"RGB: ({r}, {g}, {b})"
        color_info.config(text=rgb_str)
        color_display.config(bg=f"#{r:02x}{g:02x}{b:02x}")
    else:
        color_info.config(text="RGB: (---, ---, ---)")
        color_display.config(bg="white")


def calculate_distance(otsu_image, top_boundary, bottom_boundary):
    global continue_to_next, all_distances_df, filename
    distances = []
    total_pixels = 0
    white_pixels = 0

    # Create a polygon from the top and bottom boundaries
    polygon = top_boundary + bottom_boundary[::-1]  # Reverses the bottom boundary
    poly_path = Path(polygon)

    # Function to calculate distance to the nearest point on the top boundary
    def distance_to_top_boundary(x, y):
        return min(np.sqrt((x - bx) ** 2 + (y - by) ** 2) for bx, by in top_boundary)

    # Loop through the segmented image to find white pixels
    for y in range(otsu_image.shape[0]):
        for x in range(otsu_image.shape[1]):
            if poly_path.contains_point((x, y)):
                total_pixels += 1
                if otsu_image[y, x] == 255:  # Check if the pixel is white
                    white_pixels += 1
                    distances.append(distance_to_top_boundary(x, y))

    # Calculate the percentage of white pixels within the shape
    density_percentage = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    histogram_filename = os.path.join(
        histograms_folder_path, f"{filename}_histogram.png"
    )

    # Create and display a histogram of the distances
    if distances:
        plt.hist(distances, bins="auto")
        plt.xlabel("Distance to Nearest Top Boundary Point")
        plt.ylabel("Number of Pixels")
        plt.title("Histogram of Distances to Top Boundary")
        plt.savefig(histogram_filename)  # Save the histogram before showing it
    else:
        print("No distances to display.")

    def on_close(event):
        global top_boundary, bottom_boundary, segmented_image_path
        if segmented_image_path:  # Check if path is not None or empty
            # Ensure the image to draw on is the segmented image or a copy of the original with segmentation applied
            # Check if segmentation was applied; if not, use the original image to draw the lines
            temp_image = (
                image if segmenting_color is not None else original_image.copy()
            )

            # Draw top boundary in green
            for i in range(len(top_boundary) - 1):
                cv2.line(
                    temp_image, top_boundary[i], top_boundary[i + 1], (0, 255, 0), 2
                )  # Green line

            # Draw bottom boundary in blue
            for i in range(len(bottom_boundary) - 1):
                cv2.line(
                    temp_image,
                    bottom_boundary[i],
                    bottom_boundary[i + 1],
                    (0, 0, 255),
                    2,
                )  # Blue line

            # Create a new image to store the original and segmented images side by side
            combined_image = np.zeros(
                (
                    max(original_image.shape[0], temp_image.shape[0]),
                    original_image.shape[1] + temp_image.shape[1],
                    3,
                ),
                dtype=np.uint8,
            )

            # Copy the original image to the left side of the combined image
            combined_image[: original_image.shape[0], : original_image.shape[1]] = (
                original_image
            )

            # Copy the segmented image with boundaries to the right side of the combined image
            combined_image[: temp_image.shape[0], original_image.shape[1] :] = (
                temp_image
            )

            # Save the combined image with the original and segmented images side by side
            cv2.imwrite(segmented_image_path, combined_image)

        # Reset the boundaries for the next image
        top_boundary = []
        bottom_boundary = []

    # Reset boundaries when the plot window is closed
    plt.connect("close_event", on_close)
    plt.show()

    # Add the distances as a new column to all_distances_df
    all_distances_df.loc[:, filename] = pd.Series(distances)
    all_distances_df.loc["% Density", filename] = density_percentage
    print(f"Density for {filename}: {density_percentage}")
    continue_to_next.append(True)

    distances.sort()

"""
def calculate_percentage_distance(otsu_image, top_boundary, bottom_boundary):
    global continue_to_next, all_distances_df
    percentages = []

    # Check if the bottom boundary is not defined or is empty, we cannot calculate the percentage
    if not bottom_boundary:
        print("Bottom boundary not defined.")
        return

    # Create arrays that hold the y-coordinates of the top and bottom boundaries for each x-coordinate
    min_y_for_x = [None] * otsu_image.shape[1]
    max_y_for_x = [None] * otsu_image.shape[1]

    for x, y in top_boundary:
        if 0 <= x < otsu_image.shape[1]:
            if min_y_for_x[x] is None or y < min_y_for_x[x]:
                min_y_for_x[x] = y

    for x, y in bottom_boundary:
        if 0 <= x < otsu_image.shape[1]:
            if max_y_for_x[x] is None or y > max_y_for_x[x]:
                max_y_for_x[x] = y

    # Interpolate missing points in the boundaries
    for x in range(otsu_image.shape[1]):
        if min_y_for_x[x] is None:
            min_y_for_x[x] = min(y for y in min_y_for_x if y is not None)
        if max_y_for_x[x] is None:
            max_y_for_x[x] = max(y for y in max_y_for_x if y is not None)

    # Loop through the Otsu-segmented image to find white pixels
    for y in range(otsu_image.shape[0]):
        for x in range(otsu_image.shape[1]):
            if otsu_image[y, x] == 255:  # Check if the pixel is white
                # Calculate the percentage distance from the top boundary to the bottom boundary
                distance_top = y - min_y_for_x[x]
                total_distance = max_y_for_x[x] - min_y_for_x[x]
                if total_distance > 0:
                    percentage = (distance_top / total_distance) * 100
                    percentages.append(percentage)

    # Create a histogram of the percentages
    plt.hist(percentages, bins="auto", color="orange", alpha=0.7)
    plt.xlabel("Percentage Distance")
    plt.ylabel("Number of Pixels")
    plt.title("Histogram of Percentage Distances")
    plt.show()

    # Add the percentages as a new column to all_distances_df
    all_distances_df[filename + "_percentage"] = pd.Series(percentages)

    # If either boundary is not drawn, return without drawing the best fit line
    if not top_boundary or not bottom_boundary:
        return image

    def interpolate_points(boundary, num_points):
        # Linearly interpolate between boundary points to get the desired number of points
        return [
            boundary[int(i * (len(boundary) - 1) / (num_points - 1))]
            for i in range(num_points)
        ]

    # Get 5 equidistant points from both boundaries
    top_points = interpolate_points(top_boundary, 5)
    bottom_points = interpolate_points(bottom_boundary, 5)

    midpoints_x = [
        (top_pt[0] + bottom_pt[0]) // 2
        for top_pt, bottom_pt in zip(top_points, bottom_points)
    ]
    midpoints_y = [
        (top_pt[1] + bottom_pt[1]) // 2
        for top_pt, bottom_pt in zip(top_points, bottom_points)
    ]

    # Create weights: give higher weight to the leftmost and rightmost points.
    # For instance, a weight of 5 will make these points 5 times as significant as the other points.
    weights = (
        [5]
        + [1] * (len(midpoints_x) // 2 - 1)
        + [5]
        + [1] * (len(midpoints_x) // 2 - 1)
        + [5]
    )

    # Fit a 2nd degree polynomial to the midpoints with the specified weights
    coeff = np.polyfit(midpoints_x, midpoints_y, 2, w=weights)
    smooth_y = np.polyval(coeff, range(image.shape[1]))

    # Draw the smoothed blue line
    for i in range(1, image.shape[1]):
        cv2.line(
            image, (i - 1, int(smooth_y[i - 1])), (i, int(smooth_y[i])), (255, 0, 0), 2
        )  # Blue line
"""

def reset():
    global image, top_boundary, bottom_boundary, segmenting_color
    image = original_image.copy()
    segmenting_color = None
    segment_by_color()
    update_segmenting_color_display()


def calculate_distances_on_segmented_image():
    global image, segmenting_color
    # Check if segmenting_color is None or if the image has not been segmented
    if segmenting_color is None or np.array_equal(image, original_image):
        messagebox.showinfo(
            "Segmentation Required",
            "Please select a segmenting color and segment the image first",
        )
        return
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    calculate_distance(gray_image, top_boundary, bottom_boundary)


def draw_line(event):
    global drawing, points
    x, y = event.x, event.y
    if drawing:
        points.append((x, y))


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


def update_image(image, canvas, temp_points, color):
    temp_image = image.copy()
    for i in range(len(temp_points) - 1):
        cv2.line(temp_image, temp_points[i], temp_points[i + 1], color, 2)
    cv2image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, image=imgtk, anchor="nw")
    return imgtk


def add_statistics_to_df(df, top_boundary, bottom_boundary):
    # Calculate statistics and add them to a new DataFrame
    stats = [
        "Mean_cell_dist",
        "Median_cell_dist",
        "Mode_cell_dist",
        "Range_cell_dist",
        "Min_cell_dist",
        "Max_cell_dist",
        "Stdv_cell_dist",
    ]
    new_df = pd.DataFrame(columns=df.columns)

    for col in df.columns:
        col_data = df[col].dropna()
        if not col_data.empty:
            mean_val = col_data.mean()
            median_val = col_data.median()
            mode_val = col_data.mode()[0] if not col_data.mode().empty else "N/A"
            range_val = col_data.max() - col_data.min()
            min_val = col_data.min()
            max_val = col_data.max()
            std_val = col_data.std()
            new_df.loc["Mean_cell_dist", col] = mean_val
            new_df.loc["Median_cell_dist", col] = median_val
            new_df.loc["Mode_cell_dist", col] = mode_val
            new_df.loc["Range_cell_dist", col] = range_val
            new_df.loc["Min_cell_dist", col] = min_val
            new_df.loc["Max_cell_dist", col] = max_val
            new_df.loc["Stdv_cell_dist", col] = std_val
    # Add the "% Density" row from the original DataFrame if it exists
    if "% Density" in df.index:
        new_df.loc["% Density"] = df.loc["% Density"]

    # Add the mesh thickness statistics from the original DataFrame if they exist
    if "Min_mesh_thickness" in df.index:
        new_df.loc["Min_mesh_thickness"] = df.loc["Min_mesh_thickness"]
    if "Max_mesh_thickness" in df.index:
        new_df.loc["Max_mesh_thickness"] = df.loc["Max_mesh_thickness"]
    if "Average_mesh_thickness" in df.index:
        new_df.loc["Average_mesh_thickness"] = df.loc["Average_mesh_thickness"]

    if "% Density" in df.index:
        df_with_data_index = df.drop("% Density").reset_index(drop=True)
    else:
        df_with_data_index = df.reset_index(drop=True)

    # Add a single "Data" row index after the "% Density" row
    data_indices = [""] * len(df_with_data_index)  # Adjust the length of data_indices
    data_indices[0] = "Raw data (px distance from top)"

    df_with_data_index.index = data_indices

    # Combine the stats DataFrame and the original DataFrame
    final_df = pd.concat([new_df, df_with_data_index], axis=0)

    return final_df


def main(image_path):
    global image, original_image, continue_to_next, segmented_image_path, filename, top_boundary, bottom_boundary
    original_image = cv2.imread(image_path)
    image = original_image.copy()

    if image is None:
        print("Could not open image.")
        return

    filename = os.path.basename(image_path).split(".")[0]

    canvas.config(width=image.shape[1], height=image.shape[0])
    canvas.bind("<Motion>", update_color_display)
    update_image(image, canvas, [], (0, 0, 0))

    while True:
        try:
            if not root.winfo_exists():
                break
        except TclError:
            break

        temp_image = image.copy()

        # Drawing the boundaries and best fit line
        for i in range(len(top_boundary) - 1):
            cv2.line(temp_image, top_boundary[i], top_boundary[i + 1], (0, 255, 0), 2)
        for i in range(len(bottom_boundary) - 1):
            cv2.line(
                temp_image, bottom_boundary[i], bottom_boundary[i + 1], (0, 0, 255), 2
            )

        color = (0, 255, 0) if boundary_select.get() == 1 else (0, 0, 255)
        if drawing:
            photo = update_image(temp_image, canvas, points, color)
        else:
            photo = update_image(temp_image, canvas, [], (0, 0, 0))
        canvas.imgtk = photo

        if continue_to_next:
            break

        root.update_idletasks()
        root.update()

    # Clear continue_to_next for the next image
    continue_to_next.clear()

    cv2.destroyAllWindows()

    # Store the boundaries for the current image in the dictionary
    image_boundaries[filename] = (top_boundary.copy(), bottom_boundary.copy())

if __name__ == "__main__":
    folder_selected = filedialog.askdirectory()
    segmented_folder_path = os.path.join(folder_selected, "Segmented")
    os.makedirs(segmented_folder_path, exist_ok=True)  # Create the "Segmented" folder

    histograms_folder_path = os.path.join(folder_selected, "histograms")
    os.makedirs(histograms_folder_path, exist_ok=True)

    # Create the button frame and pack it at the top
    button_frame = Frame(root)
    button_frame.pack(side=TOP, fill=BOTH)

    # Buttons setup
    top_button = Radiobutton(
        button_frame, text="Top boundary", variable=boundary_select, value=1
    )
    top_button.pack(side=LEFT)
    bottom_button = Radiobutton(
        button_frame, text="Bottom boundary", variable=boundary_select, value=2
    )
    bottom_button.pack(side=LEFT)
    confirm_button = Button(
        button_frame, text="Confirm", command=calculate_distances_on_segmented_image
    )
    confirm_button.pack(side=RIGHT)
    reset_button = Button(button_frame, text="Reset", command=reset)
    reset_button.pack(side=RIGHT)
    segment_button = Button(button_frame, text="Segment", command=segment_by_color)
    segment_button.pack(side=RIGHT)

    color_info = Label(button_frame, text="RGB: (---, ---, ---)")
    color_info.pack(side=LEFT)

    color_display = Label(button_frame, width=3, bg="white")
    color_display.pack(side=LEFT)

    # Create and place the segmenting color label and display
    segmenting_color_label = Label(button_frame, text="Segmenting Color: None")
    segmenting_color_label.pack(side=LEFT)

    segmenting_color_display = Label(button_frame, width=3, bg="white")
    segmenting_color_display.pack(side=LEFT)

    # Now pack the canvas
    canvas = Canvas(root)
    canvas.pack(side=TOP)

    canvas.bind("<ButtonPress-1>", start_draw)
    canvas.bind("<ButtonRelease-1>", stop_draw)
    canvas.bind("<B1-Motion>", draw_line)
    canvas.bind("<ButtonPress-3>", save_segmenting_color)  # Right-click to select color

    for filename in os.listdir(folder_selected):
        if filename.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            filepath = os.path.join(folder_selected, filename)
            segmented_image_path = os.path.join(
                segmented_folder_path, filename
            )  # Path to save segmented image
            print(f"Processing {filename}...")
            main(filepath)

    # Retrieve the boundaries for the last processed image from the dictionary
    last_filename = os.path.basename(filepath).split(".")[0]
    top_boundary, bottom_boundary = image_boundaries.get(last_filename, ([], []))

    print("Final top boundary:", top_boundary)
    print("Final bottom boundary:", bottom_boundary)

    # Add statistics to the DataFrame
    all_distances_df_with_stats = add_statistics_to_df(
        all_distances_df, top_boundary, bottom_boundary
    )

    # Save the DataFrame with statistics to Excel
    all_distances_df_with_stats.to_excel(
        os.path.join(folder_selected, "all_distances.xlsx"), index=True
    )