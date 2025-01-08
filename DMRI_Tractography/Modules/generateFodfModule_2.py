import os
import sys
import time
import shutil
from pathlib import Path
import paramiko
import slicer

sibling_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scilpy'))

# Add it to sys.path
if sibling_folder_path not in sys.path:
    sys.path.append(sibling_folder_path)

from scripts.scil_frf_ssst import main as scil_frf_ssst_main
from scripts.scil_fodf_ssst import main as scil_fodf_ssst_main
from scripts.scil_dti_metrics import main as scil_dti_metrics_main
from scripts.scil_fodf_metrics import main as scil_fodf_metrics_main


class SyncthingTractographyProcessor:
    def __init__(self, white_mask, diffusion, bvals, bvecs, output_dir):
        self.local_input_dir = Path("/Users/anoushkritgoel/Github/SlicerTracto/Input")
        self.local_output_dir = Path("/Users/anoushkritgoel/Github/SlicerTracto/Output")
        self.ssh_key_path = "/Users/anoushkritgoel/.ssh/mahirj_param"

        self.remote_input_dir = Path("/scratch/mahirj.scee.iitmandi/DMRI_TRACTOGRAPHY/Input")
        self.remote_output_dir = Path("/scratch/mahirj.scee.iitmandi/DMRI_TRACTOGRAPHY/Output")
        self.remote_script_path = Path("/scratch/mahirj.scee.iitmandi/DMRI_TRACTOGRAPHY/Server/fODFgen.py")

        self.white_mask = white_mask
        self.diffusion = diffusion
        self.bvals = bvals
        self.bvecs = bvecs
        self.output_dir = output_dir

        # SSH Configuration
        self.remote_host = "paramhimalaya.iitmandi.ac.in"
        self.remote_port = 4422
        self.username = "mahirj.scee.iitmandi"
        # self.ssh_key_path = os.path.expanduser("~/.ssh/filename")

        self.ssh_client = None
    
    def connect_ssh(self):
        """Establish an SSH connection to the remote server."""
        slicer.util.showStatusMessage("Connecting to remote server...")
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            private_key = paramiko.RSAKey(filename=self.ssh_key_path)
            self.ssh_client.connect(
                hostname=self.remote_host,
                port=self.remote_port,
                username=self.username,
                pkey=private_key
            )
            slicer.util.showStatusMessage("SSH connection established.")
        except Exception as e:
            slicer.util.showStatusMessage(f"SSH connection failed: {str(e)}")
            raise
    
    def copy_files_to_local_input(self):
        """Copy input files to the local input directory."""
        self.local_input_dir.mkdir(parents=True, exist_ok=True)
        files = [self.white_mask, self.diffusion, self.bvals, self.bvecs]
        for file in files:
            destination = self.local_input_dir / Path(file).name
            shutil.copy(file, destination)
            print(f"Copied {file} to {destination}")

    def wait_for_sync(self, target_dir, target_file, timeout=3600):
        """Wait for a file to appear in the specified directory within the timeout."""
        start_time = time.time()
        target_path = target_dir / target_file
        while not target_path.exists():
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Synchronization timed out for {target_path}")
            time.sleep(2)
        print(f"File {target_path} is now available.")
    

    def execute_remote_script(self):
        """Run the FODF pipeline script on the remote server."""
        print("Executing FODF pipeline on remote server...")

        self.white_mask = os.path.basename(self.white_mask)
        self.diffusion = os.path.basename(self.diffusion)
        self.bvals = os.path.basename(self.bvals)
        self.bvecs = os.path.basename(self.bvecs)

        conda_env_name = "slicer_env"
        command = (
            f"source ~/.bashrc && "
            f"conda activate {conda_env_name} && "
            f"python3 {self.remote_script_path} "
            f"--white_mask {self.remote_input_dir / Path(self.white_mask).name} "
            f"--diffusion {self.remote_input_dir / Path(self.diffusion).name} "
            f"--bvals {self.remote_input_dir / Path(self.bvals).name} "
            f"--bvecs {self.remote_input_dir / Path(self.bvecs).name} "
            f"--output {self.remote_output_dir / Path(self.output_dir).name}"
        )

        stdin, stdout, stderr = self.ssh_client.exec_command(command, get_pty=True)

        while True:
            line = stdout.readline()
            if not line and stdout.channel.exit_status_ready():
                break
            print(line.strip())  # Print script output

        error = stderr.read().decode('utf-8')
        if error:
            raise RuntimeError(f"Error during remote execution: {error}")

        print("Remote execution completed.")
    
    def ensure_output_sync(self, sync_timeout=3600):
        """Wait for the output file to be generated remotely and then synced locally."""
        remote_output_file = Path(self.output_dir).name

        # Wait for the file to appear in the remote directory
        print("Waiting for remote output generation...")
        self.wait_for_sync(self.remote_output_dir, remote_output_file, timeout=sync_timeout)

        # Wait for the file to sync to the local directory
        print("Waiting for output synchronization to local...")
        self.wait_for_sync(self.local_output_dir, remote_output_file, timeout=sync_timeout)


    def run_pipeline(self):
        """Run the complete FODF processing pipeline."""
        try:
            start_time = time.time()

            # Connect to SSH
            self.connect_ssh()

            # Copy input files to local input directory
            self.copy_files_to_local_input()

            # Wait for synchronization to remote input directory
            for file in [self.white_mask, self.diffusion, self.bvals, self.bvecs]:
                self.wait_for_sync(self.remote_input_dir, Path(file).name)

            # Execute remote script
            self.execute_remote_script()

            # Wait for output synchronization
            self.ensure_output_sync()

            elapsed_time = time.time() - start_time
            print(f"Pipeline completed in {elapsed_time:.2f} seconds.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if self.ssh_client:
                self.ssh_client.close()
                slicer.util.showStatusMessage("SSH connection closed.")
                
                
class GenerateFODF:
    def __init__(self):
        self.whiteMaskBiftiPath : str = None
        self.diffusionNiftiPath : str = None
        self.bvalsPath : str = None
        self.bvecsPath : str = None
        self.fodfPath : str = None
    
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
        
        print("Generating fODF using remote connection")
        
        white_mask = self.whiteMaskBiftiPath
        diffusion = self.diffusionNiftiPath
        bvals = self.bvalsPath
        bvecs = self.bvecsPath
        fodfPath = self.fodfPath

        processor = SyncthingTractographyProcessor(
            white_mask=white_mask,
            diffusion=diffusion,
            bvals=bvals,
            bvecs=bvecs,
            output_dir=fodfPath
        )

        processor.run_pipeline()

    def saveFodf(self):
        print("Save FODF")
    
    def visualizeFodf(self):
        print("Visualize FODF")

