import os
from tkinter import filedialog, messagebox
import pandas as pd
import cv2
from gui import create_gui, root, canvas, update_image
from utils import process_image, add_statistics_to_df
import gui

# Global variables
image_boundaries = {}
all_distances_df = pd.DataFrame()

def main():
    global image_boundaries, all_distances_df

    folder_selected = filedialog.askdirectory()
    if not folder_selected:
        print("No folder selected. Exiting.")
        return

    segmented_folder_path = os.path.join(folder_selected, "Segmented")
    os.makedirs(segmented_folder_path, exist_ok=True)

    histograms_folder_path = os.path.join(folder_selected, "histograms")
    os.makedirs(histograms_folder_path, exist_ok=True)

    create_gui()

    for filename in os.listdir(folder_selected):
        if filename.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            filepath = os.path.join(folder_selected, filename)
            segmented_image_path = os.path.join(segmented_folder_path, filename)
            print(f"Processing {filename}...")
            top_boundary, bottom_boundary = process_image(filepath, segmented_image_path, canvas, update_image)

            # Store the boundaries for the current image
            image_boundaries[filename] = (top_boundary, bottom_boundary)

    # Retrieve the boundaries for the last processed image
    if image_boundaries:
        last_filename = list(image_boundaries.keys())[-1]
        top_boundary, bottom_boundary = image_boundaries[last_filename]
    else:
        top_boundary, bottom_boundary = [], []

    # Add statistics to the DataFrame
    all_distances_df_with_stats = add_statistics_to_df(all_distances_df, top_boundary, bottom_boundary)

    # Save the DataFrame with statistics to Excel
    all_distances_df_with_stats.to_excel(os.path.join(folder_selected, "all_distances.xlsx"), index=True)

if __name__ == "__main__":
    main()
    root.mainloop()