# Cell Infiltration Analysis Tool

## Overview

This program is designed to analyze cell infiltration in tissue samples. It allows users to define the boundaries of a tissue sample, select cells of interest based on color, and calculate the extent of cell infiltration. The tool provides visual outputs and detailed statistical data for further analysis.

## Features

- Load and process multiple image files (PNG, JPG, JPEG, TIF, TIFF)
- Interactive drawing of top and bottom tissue boundaries
- Color-based cell segmentation
- Calculation of cell distances from the top boundary
- Generation of histograms showing cell distribution
- Output of comprehensive statistics in Excel format
- Side-by-side visualization of original and segmented images

## Requirements

- Python 3.x
- Required libraries: tkinter, PIL, OpenCV (cv2), numpy, matplotlib, pandas

## Usage

1. Run the script and select the folder containing your tissue sample images.
2. For each image:
   a. Draw the top boundary (green) and bottom boundary (blue) of the tissue.
   b. Right-click to select the color of the cells you want to analyze.
   c. Click "Segment" to isolate the cells based on the selected color.
   d. Click "Confirm" to calculate distances and generate the histogram.
3. After processing all images, the program will save:
   - Segmented images in a "Segmented" subfolder
   - Histograms in a "histograms" subfolder
   - An Excel file named "all_distances.xlsx" with comprehensive statistics

## Output

- Segmented images: Original and analyzed images side by side
- Histograms: Distribution of cell distances from the top boundary
- Excel file: Contains raw distance data and calculated statistics including:
  - Mean, median, mode, range, min, max, and standard deviation of cell distances
  - Density percentage of cells within the tissue
  - Mesh thickness statistics (if applicable)

## Notes

- The "Reset" button clears current segmentation and boundaries.
- Right-click to select a new segmenting color at any time.
- Ensure proper contrast between tissue and cells for best results.

## Troubleshooting

If you encounter any issues:
- Check that all required libraries are installed
- Ensure input images are in supported formats
- Verify that you have write permissions in the selected folder

For further assistance, please contact phillip.baekk@gmail.com