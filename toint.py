import sys
import nibabel as nib
import numpy as np

def create_binary_mask(input_path, output_path, threshold=0.5):
    # Load the image
    img = nib.load(input_path)

    # Get the image data
    data = img.get_fdata()

    # Threshold the data to create a binary mask
    mask_data = (data > threshold).astype(np.uint8)

    # Create a new nibabel image with the binary mask
    mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
    mask_img.set_data_dtype(np.uint8)

    # Save the new image
    nib.save(mask_img, output_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_image.nii.gz output_image.nii.gz")
        sys.exit(1)

    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]

    create_binary_mask(input_image_path, output_image_path)
