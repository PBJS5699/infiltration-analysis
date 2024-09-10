import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import messagebox
import os
from matplotlib.path import Path

def segment_by_color(image, original_image, segmenting_color):
    if segmenting_color is None:
        return original_image.copy()
    else:
        range = 40
        color_lower = np.array([max(segmenting_color[2] - range, 0), max(segmenting_color[1] - range, 0), max(segmenting_color[0] - range, 0)], dtype="uint8")
        color_upper = np.array([min(segmenting_color[2] + range, 255), min(segmenting_color[1] + range, 255), min(segmenting_color[0] + range, 255)], dtype="uint8")
        mask = cv2.inRange(image, color_lower, color_upper)
        _, otsu_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        segmented_image = cv2.cvtColor(otsu_mask, cv2.COLOR_GRAY2BGR)
        return segmented_image

def calculate_distance(otsu_image, top_boundary, bottom_boundary, filename, all_distances_df, histograms_folder_path):
    distances = []
    total_pixels = 0
    white_pixels = 0

    polygon = top_boundary + bottom_boundary[::-1]
    poly_path = Path(polygon)

    def distance_to_top_boundary(x, y):
        return min(np.sqrt((x - bx) ** 2 + (y - by) ** 2) for bx, by in top_boundary)

    for y in range(otsu_image.shape[0]):
        for x in range(otsu_image.shape[1]):
            if poly_path.contains_point((x, y)):
                total_pixels += 1
                if otsu_image[y, x] == 255:
                    white_pixels += 1
                    distances.append(distance_to_top_boundary(x, y))

    density_percentage = (white_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    histogram_filename = os.path.join(histograms_folder_path, f"{filename}_histogram.png")

    if distances:
        plt.hist(distances, bins="auto")
        plt.xlabel("Distance to Nearest Top Boundary Point")
        plt.ylabel("Number of Pixels")
        plt.title("Histogram of Distances to Top Boundary")
        plt.savefig(histogram_filename)
    else:
        print("No distances to display.")

    all_distances_df.loc[:, filename] = pd.Series(distances)
    all_distances_df.loc["% Density", filename] = density_percentage
    print(f"Density for {filename}: {density_percentage}")

    return all_distances_df

def add_statistics_to_df(df, top_boundary, bottom_boundary):
    stats = ["Mean_cell_dist", "Median_cell_dist", "Mode_cell_dist", "Range_cell_dist", "Min_cell_dist", "Max_cell_dist", "Stdv_cell_dist"]
    new_df = pd.DataFrame(columns=df.columns)

    for col in df.columns:
        col_data = df[col].dropna()
        if not col_data.empty:
            new_df.loc["Mean_cell_dist", col] = col_data.mean()
            new_df.loc["Median_cell_dist", col] = col_data.median()
            new_df.loc["Mode_cell_dist", col] = col_data.mode()[0] if not col_data.mode().empty else "N/A"
            new_df.loc["Range_cell_dist", col] = col_data.max() - col_data.min()
            new_df.loc["Min_cell_dist", col] = col_data.min()
            new_df.loc["Max_cell_dist", col] = col_data.max()
            new_df.loc["Stdv_cell_dist", col] = col_data.std()

    if "% Density" in df.index:
        new_df.loc["% Density"] = df.loc["% Density"]

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

    data_indices = [""] * len(df_with_data_index)
    data_indices[0] = "Raw data (px distance from top)"

    df_with_data_index.index = data_indices

    final_df = pd.concat([new_df, df_with_data_index], axis=0)

    return final_df

def process_image(image_path, segmented_image_path, canvas, update_image):
    from gui import boundary_select, drawing, points, top_boundary, bottom_boundary
    
    original_image = cv2.imread(image_path)
    image = original_image.copy()

    if image is None:
        print("Could not open image.")
        return

    filename = os.path.basename(image_path).split(".")[0]

    canvas.config(width=image.shape[1], height=image.shape[0])
    update_image(image, canvas, [], (0, 0, 0))

    continue_to_next = []

    while True:
        try:
            if not canvas.winfo_exists():
                break
        except:
            break

        temp_image = image.copy()

        for i in range(len(top_boundary) - 1):
            cv2.line(temp_image, top_boundary[i], top_boundary[i + 1], (0, 255, 0), 2)
        for i in range(len(bottom_boundary) - 1):
            cv2.line(temp_image, bottom_boundary[i], bottom_boundary[i + 1], (0, 0, 255), 2)

        color = (0, 255, 0) if boundary_select.get() == 1 else (0, 0, 255)
        if drawing:
            photo = update_image(temp_image, canvas, points, color)
        else:
            photo = update_image(temp_image, canvas, [], (0, 0, 0))
        canvas.imgtk = photo

        if continue_to_next:
            break

        canvas.update_idletasks()
        canvas.update()

    cv2.destroyAllWindows()

    return top_boundary.copy(), bottom_boundary.copy()