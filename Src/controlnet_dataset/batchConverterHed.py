import os
import subprocess

input_dir = 'images'
output_dir = 'imagesControlhed'
model = 'bsds500'  # You can change this if needed

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all image files from the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '_hed.jpg')

        # Run the HED script using subprocess
        subprocess.run([
            'python', 'imagesToHed.py',
            '--model=' + model,
            '--in=' + input_path,
            '--out=' + output_path
        ])