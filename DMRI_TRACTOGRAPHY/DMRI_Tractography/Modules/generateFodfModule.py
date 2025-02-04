import os
import sys

sibling_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scilpy'))

# Add it to sys.path
if sibling_folder_path not in sys.path:
    sys.path.append(sibling_folder_path)

from scripts.scil_frf_ssst import main as scil_frf_ssst_main
from scripts.scil_fodf_ssst import main as scil_fodf_ssst_main
from scripts.scil_dti_metrics import main as scil_dti_metrics_main
from scripts.scil_fodf_metrics import main as scil_fodf_metrics_main

class GenerateFODF:
    def __init__(self):
        self.whiteMaskBiftiPath : str = None
        self.diffusionNiftiPath : str = None
        self.bvalsPath : str = None
        self.bvecsPath : str = None
        self.fodfPath : str = None
        self.outputText = None

    
    @staticmethod
    def isValidPath(path: str) -> bool:
        """Validate if the provided path exists and is a file."""
        return os.path.isfile(path)

    # Setter functions with validation
    def setWhiteMaskBiftiPath(self, path: str):
        """Set the path for white mask after validation."""
        if self.isValidPath(path):
            self.whiteMaskBiftiPath = path
            print(f"White mask path set to: {self.whiteMaskBiftiPath}")
        else:
            print(f"Invalid white mask path: {path}")

    def setDiffusionNiftiPath(self, path: str):
        """Set the path for diffusion NIFTI file after validation."""
        if self.isValidPath(path):
            self.diffusionNiftiPath = path
            print(f"Diffusion NIFTI path set to: {self.diffusionNiftiPath}")
        else:
            print(f"Invalid diffusion NIFTI path: {path}")

    def setBvalsPath(self, path: str):
        """Set the path for self.bvalsPath file after validation."""
        if self.isValidPath(path):
            self.bvalsPath = path
            print(f"Bvals path set to: {self.bvalsPath}")
        else:
            print(f"Invalid self.bvalsPath path: {path}")

    def setBvecsPath(self, path: str):
        """Set the path for self.bvecsPath file after validation."""
        if self.isValidPath(path):
            self.bvecsPath = path
            print(f"Bvecs path set to: {self.bvecsPath}")
        else:
            print(f"Invalid self.bvecsPath path: {path}")
    
    def setFodfPath(self, path: str):
        if self.isValidPath(path):
            self.fodfPath = path
            print(f"Fodf path set to: {self.fodfPath}")
        else:
            print(f"Invalid Fodf path: {path}")
    
    def generateFodf(self):
        preproc_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
        
        try:
            # Ensure output directory exists
            os.makedirs(preproc_data_dir, exist_ok=True)

            # Step 2: Estimate response function using SSST
            frf_file = os.path.join(preproc_data_dir, "frf.txt")
            frf_args = [
                "scil_frf_ssst.py", self.diffusionNiftiPath, self.bvalsPath, self.bvecsPath, frf_file,
                "--mask_wm", self.whiteMaskBiftiPath, "-f", "-v"
            ]
            sys.argv = frf_args
            scil_frf_ssst_main()
            print("Response function estimated successfully.")

            # Step 3: Generate FODF using SSST
            fodf_file = os.path.join(preproc_data_dir, "fodf.nii.gz")
            fodf_args = [
                "scil_fodf_ssst.py", self.diffusionNiftiPath, self.bvalsPath, self.bvecsPath, frf_file, fodf_file,
                "--processes", "8", "--sh_order", "6", "-f"
            ]
            sys.argv = fodf_args
            scil_fodf_ssst_main()
            print("FODF generated successfully.")

            # Compute DTI metrics
            dti_args = ["scil_dti_metrics.py", self.diffusionNiftiPath, self.bvalsPath, self.bvecsPath]
            sys.argv = dti_args
            scil_dti_metrics_main()
            print("DTI metrics computed successfully.")

            # Compute FODF metrics
            fodf_metrics_args = ["scil_fodf_metrics.py", fodf_file, "-f"]
            sys.argv = fodf_metrics_args
            scil_fodf_metrics_main()
            print("FODF metrics computed successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            # Reset sys.argv to avoid affecting other parts of the program
            sys.argv = ["main.py"]

    def saveFodf(self):
        print("Save FODF")
    
    def visualizeFodf(self):
        print("Visualize FODF")


    

