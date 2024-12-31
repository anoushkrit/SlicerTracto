import os
import time
from pathlib import Path
import paramiko

class RemoteHCPScoring:
    def __init__(self, predicted_trk, gt_trk, output_dir):
        
        self.local_input_dir = Path("/Users/mahir/Desktop/MTP/SlicerTracto-extension-with-integrated-modules/DMRI_TRACTOGRAPHY/Input")
        self.local_output_dir = Path("/Users/mahir/Desktop/MTP/SlicerTracto-extension-with-integrated-modules/DMRI_TRACTOGRAPHY/Output")

        self.remote_input_dir = Path("/scratch/mahirj.scee.iitmandi/DMRI_TRACTOGRAPHY/Input")
        self.remote_output_dir = Path("/scratch/mahirj.scee.iitmandi/DMRI_TRACTOGRAPHY/Output")
        self.remote_script_path = Path("/scratch/mahirj.scee.iitmandi/DMRI_TRACTOGRAPHY/Server/scoring.py")

        self.predicted_trk = predicted_trk
        self.gt_trk = gt_trk
        self.output_dir = output_dir

        # SSH Configuration
        self.remote_host = "paramhimalaya.iitmandi.ac.in"
        self.remote_port = 4422
        self.username = "mahirj.scee.iitmandi"
        self.ssh_key_path = os.path.expanduser("~/.ssh/filename")
        self.ssh_client = None

    def connect_ssh(self):
        """Establish an SSH connection to the remote server."""
        print("Connecting to remote server...")
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
            print("SSH connection established.")
        except Exception as e:
            print(f"SSH connection failed: {str(e)}")
            raise


    def execute_remote_script(self):
        """Run the HCP scoring script on the remote server."""
        print("Executing HCP scoring script on remote server...")
        
        conda_env_name = "slicer_env"
        
        command = (
            f"source ~/.bashrc && "
            f"conda activate {conda_env_name} && " 
            f"python3 {self.remote_script_path} "
            f"--predicted_trk {self.remote_input_dir / Path(self.predicted_trk).name} "
            f"--gt_trk {self.remote_input_dir / Path(self.gt_trk).name}"
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
        """Run the complete HCP scoring pipeline."""
        try:
            start_time = time.time()

            # Connect to SSH
            self.connect_ssh()

            # Execute remote script
            self.execute_remote_script()

            # Ensure output synchronization
            self.ensure_output_sync()

            elapsed_time = time.time() - start_time
            print(f"Pipeline completed in {elapsed_time:.2f} seconds.")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if self.ssh_client:
                self.ssh_client.close()
                print("SSH connection closed.")

if __name__ == "__main__":
    processor = RemoteHCPScoring(
        predicted_trk="/path/to/local/predicted.trk",
        gt_trk="/path/to/local/gt.trk",
        output_dir="hcp_scores_output.json"  # Example output filename
    )
    processor.run_pipeline()
