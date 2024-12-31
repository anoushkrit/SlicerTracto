import os
import sys
import time
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
        self.local_input_dir = Path("./DMRI_TRACTOGRAPHY/Input")
        self.local_output_dir = Path("./DMRI_TRACTOGRAPHY/Output")

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
        self.ssh_key_path = os.path.expanduser("~/.ssh/filename")
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
    

    def execute_remote_script(self):
        """Run the FODF pipeline script on the remote server."""
        print("Executing FODF pipeline on remote server...")

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
        print("Waiting for remote output generation...")
        start_time = time.time()

        # Check if the output file exists on the remote server
        remote_output_file = self.remote_output_dir / self.output_file
        while True:
            stdin, stdout, stderr = self.ssh_client.exec_command(f"ls {remote_output_file}")
            if stdout.read().decode().strip():
                print(f"Remote output file {remote_output_file} is now available.")
                break
            if time.time() - start_time > sync_timeout:
                raise TimeoutError("Remote output generation timed out.")
            time.sleep(2)

        print("Waiting for output synchronization to local...")
        start_time = time.time()
        local_output_file = self.local_output_dir / self.output_file

        # Wait for the file to sync locally
        while not local_output_file.exists():
            if time.time() - start_time > sync_timeout:
                raise TimeoutError("Output synchronization timed out.")
            time.sleep(2)

        print(f"Output file {local_output_file} is now available locally.")


    def run_pipeline(self):
        """Run the complete FODF processing pipeline."""
        try:
            start_time = time.time()

            # Connect to SSH
            self.connect_ssh()

            # Upload input files
            self.upload_files()

            # Execute remote script
            self.execute_remote_script()

            # Download results
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
        
        white_mask = self.whiteMaskBiftiPath
        diffusion = self.diffusionNiftiPath
        bvals = self.bvalsPath
        bvecs = self.bvecsPath
        fodfPath = self.fodfPath

        processor = SyncthingTractographyProcessor(
            seeding_mask_file=os.path.basename(white_mask),
            diffusion=os.path.basename(diffusion),
            bvals=os.path.basename(bvals),
            bvecs=os.path.basename(bvecs),
            output_dir=fodfPath
        )

        processor.run_pipeline()

    def saveFodf(self):
        print("Save FODF")
    
    def visualizeFodf(self):
        print("Visualize FODF")

