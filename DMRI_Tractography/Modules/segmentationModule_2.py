import vtk
import os
import sys
import time
from pathlib import Path
import paramiko

import slicer
import random

from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles
from dipy.io.streamline import save_trk
from dipy.tracking.streamline import transform_streamlines
from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamline import Streamlines


DEFAULT_SEGMENTED_TRK_FILE_NAME_PREFIX = 'segmentedTrk'
file_path = os.path.abspath(__file__)
DEFAULT_DIR = os.path.dirname(file_path)

class SyncthingProcessor:
    def __init__(self, input_trk, output_dir, threshold):
        self.local_input_dir = Path("/Users/mahir/Desktop/MTP/SlicerTracto-extension-with-integrated-modules/DMRI_TRACTOGRAPHY/Input")
        self.local_output_dir = Path("/Users/mahir/Desktop/MTP/SlicerTracto-extension-with-integrated-modules/DMRI_TRACTOGRAPHY/Output")

        self.remote_input_dir = Path("/scratch/mahirj.scee.iitmandi/DMRI_TRACTOGRAPHY/Input")
        self.remote_output_dir = Path("/scratch/mahirj.scee.iitmandi/DMRI_TRACTOGRAPHY/Output")
        self.remote_script_path = Path("/scratch/mahirj.scee.iitmandi/DMRI_TRACTOGRAPHY/Server/segmenting.py")

        self.input_trk = input_trk
        self.output_dir = output_dir
        self.threshold = threshold

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
    
    def verify_input_files(self):
        """Verify that required input files exist locally."""
        slicer.util.showStatusMessage("Verifying input files...")
        for file in [self.seeding_mask_file, self.fodf_file]:
            if not Path(self.remote_input_dir / file).exists():
                raise FileNotFoundError(f"Required input file not found: {file}")
        slicer.util.showStatusMessage("All input files verified.")

    def execute_remote_script(self):
        """Run the tractography script on the remote server."""
        slicer.util.showStatusMessage("Executing remote script...")

        conda_env_name = "slicer_env"

        command = (
            f"source ~/.bashrc && "
            f"conda activate {conda_env_name} && "
            f"python3 {self.remote_script_path} "
            f"--input_trk {self.remote_input_dir / Path(self.input_trk).name} "
            f"--output_dir {self.remote_output_dir / Path(self.output_dir).name} "
            f"--threshold {self.threshold}"
        )


        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command)

            while True:
                line = stdout.readline()
                if not line and stdout.channel.exit_status_ready():
                    break
                print(line.strip())  # Print output from the remote script

            error = stderr.read().decode('utf-8')
            if error:
                raise RuntimeError(f"Error during remote execution: {error}")

            slicer.util.showStatusMessage("Remote execution completed.")
        except Exception as e:
            slicer.util.showStatusMessage(f"Execution failed: {e}")
            raise
    
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
        """Run the complete pipeline."""
        try:
            start_time = time.time()

            # Connect to SSH
            self.connect_ssh()

            # Execute remote script
            self.execute_remote_script()

            # Ensure output synchronization
            # self.ensure_output_sync()

            elapsed_time = time.time() - start_time
            slicer.util.showStatusMessage(f"Pipeline completed in {elapsed_time:.2f} seconds.")
        except Exception as e:
            slicer.util.errorDisplay(f"An error occurred: {e}")
        finally:
            if self.ssh_client:
                self.ssh_client.close()
                slicer.util.showStatusMessage("SSH connection closed.")
                
class Segmentation:
    def __init__(self):
        self.trkPath: str = None
        self.segmentedTrkFolderPath: str = DEFAULT_DIR
        self.outputText = None
        self.clusters = None
        self.threshold: float = 10.0

    @staticmethod
    def _isValidPath(path: str) -> bool:
        """Validate if the provided path exists and is a file."""
        return os.path.isfile(path) or os.path.isdir(path)

    def set_trkPath(self, path: str):
        """Set the path for the TRK file after validation."""
        if self._isValidPath(path):
            self.trkPath = path
            print(f"TRK path set to: {self.trkPath}")
        else:
            raise FileNotFoundError(f"Invalid TRK path: {path}")

    def set_segmentedTrkFolderPath(self, path: str):
        """Set the path for the segmented TRK folder after validation."""
        if self._isValidPath(path):
            self.segmentedTrkFolderPath = path
            print(f"Segmented TRK folder path set to: {self.segmentedTrkFolderPath}")
        else:
            raise FileNotFoundError(f"Invalid segmented TRK folder path: {path}")

    def set_threshold(self, threshold):
        self.threshold = float(threshold)
        print(f"Threshold value set to: {self.threshold}")

    def segmentTrk(self):
        print("Segmenting using remote ssh compute")
        
        """Segment the TRK file using remote script execution."""
        if not self.trkPath:
            raise ValueError("TRK file path is not set.")
        if not self.segmentedTrkFolderPath:
            raise ValueError("Segmented TRK folder path is not set.")

        try:
            print("Starting segmentation process...")
            tractogram = load_tractogram(self.trkPath, reference="same", bbox_valid_check=False)
            streamlines = Streamlines(tractogram.streamlines)
            # streamlines = tractogram.streamlines
            qb = QuickBundles(threshold=self.threshold)
            self.clusters = qb.cluster(streamlines)
           
            # Initialize SSH Processor
            processor = SyncthingProcessor(
                input_trk=self.trkPath,
                output_dir=self.segmentedTrkFolderPath,
                threshold=self.threshold,
            )

            # Run pipeline (includes remote execution and output synchronization)
            processor.run_pipeline()

            print("\nSegmentation process completed successfully.")

        except Exception as e:
            slicer.util.errorDisplay(f"Error during segmentation: {e}")

    def visualizeSegmentation(self):
        """Visualize clustering results using VTK in Slicer3D."""
        if not self.clusters:
            slicer.util.warnign("\nNo clusters to visualize. Please run `segmentTrk` first.")
            return

        try:
            print("\nVisualizing clusters...")
            colors = self._generateColors(len(self.clusters))

            for idx, cluster in enumerate(self.clusters):
                vtk_polydata = self._convertToVTK(cluster)

                # Create a new model node
                modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"Cluster {idx + 1}")
                modelNode.SetAndObservePolyData(vtk_polydata)

                # Add a display node explicitly
                displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
                modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())

                # Set color and visibility
                displayNode.SetColor(colors[idx])
                displayNode.SetVisibility(True)

            print("\nClusters visualized successfully.")
        except Exception as e:
            slicer.util.errorDisplay(f"Error during visualization: {e}")

    def _convertToVTK(self, cluster):
        """Convert a cluster of streamlines to VTK polydata."""
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        for streamline in cluster:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(streamline))

            for i, point in enumerate(streamline):
                pointId = points.InsertNextPoint(point)
                line.GetPointIds().SetId(i, pointId)

            lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        return polydata

    def _generateColors(self, numClusters):
        """Generate distinct colors for each cluster."""
        colors = []
        colormap = vtk.vtkNamedColors()
        predefinedColors = list(colormap.GetColorNames())
        for i in range(numClusters):
            colorName = predefinedColors[i % len(predefinedColors)]
            colors.append(colormap.GetColor3d(colorName))
        return colors
