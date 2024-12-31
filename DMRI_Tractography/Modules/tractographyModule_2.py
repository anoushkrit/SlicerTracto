import os
import sys
import time
from pathlib import Path
import paramiko
import slicer
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from vtk import vtkPolyDataReader
import vtk


# Get the path to the scilpy folder and add to sys paths
sibling_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scilpy'))
if sibling_folder_path not in sys.path:
    sys.path.append(sibling_folder_path)

DEFAULT_TRK_FILE_NAME = "result.trk"
DEFAULT_VTK_FILE_NAME = "result.vtk"

class SyncthingTractographyProcessor:
    def __init__(self, seeding_mask_file, fodf_file, output_file, step_size, algotype):
        self.local_input_dir = Path("./DMRI_TRACTOGRAPHY/Input")
        self.local_output_dir = Path("./DMRI_TRACTOGRAPHY/Output")

        self.remote_input_dir = Path("/scratch/mahirj.scee.iitmandi/DMRI_TRACTOGRAPHY/Input")
        self.remote_output_dir = Path("/scratch/mahirj.scee.iitmandi/DMRI_TRACTOGRAPHY/Output")
        self.remote_script_path = Path("/scratch/mahirj.scee.iitmandi/DMRI_TRACTOGRAPHY/Server/genTrk.py")

        self.seeding_mask_file = seeding_mask_file
        self.fodf_file = fodf_file
        self.output_file = output_file

        self.step_size = step_size
        self.algotype = algotype

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
            f"--approx_mask_path {self.remote_input_dir / self.seeding_mask_file} "
            f"--fodf_path {self.remote_input_dir / self.fodf_file} "
            f"--output_dir {self.remote_output_dir} "
            f"--step_size {self.step_size} "
            f"--algo {self.algotype}"
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
            self.ensure_output_sync()

            elapsed_time = time.time() - start_time
            slicer.util.showStatusMessage(f"Pipeline completed in {elapsed_time:.2f} seconds.")
        except Exception as e:
            slicer.util.errorDisplay(f"An error occurred: {e}")
        finally:
            if self.ssh_client:
                self.ssh_client.close()
                slicer.util.showStatusMessage("SSH connection closed.")

class Tractography:
    def __init__(self):
        self.approxMaskPath: str = None
        self.fodfPath: str = None
        self.stepSize: float = 0.2
        self.algo: str = "det"
        self.trkPath : str = None
        self.outputText = None
        self.output_trk_path = "/Users/mahir/Desktop/MTP/SlicerTracto-extension-with-integrated-modules/DMRI_TRACTOGRAPHY/Output/result.trk"

    @staticmethod
    def _isValidPath(path: str) -> bool:
        """Validate if the provided path exists and is a file."""
        return os.path.isfile(path) or os.path.isdir(path)

    # Updated setter methods with validation
    def set_approxMaskPath(self, path: str):
        """Set the path for the approximate mask after validation."""
        if self._isValidPath(path):
            self.approxMaskPath = path
            print(f"Approximate mask path set to: {self.approxMaskPath}")
        else:
            raise FileNotFoundError(f"Invalid approximate mask path: {path}")

    def set_fodfPath(self, path: str):
        """Set the path for the FODF file after validation."""
        if self._isValidPath(path):
            self.fodfPath = path
            print(f"FODF path set to: {self.fodfPath}")
        else:
            raise FileNotFoundError(f"Invalid FODF path: {path}")

    def set_trkPath(self, path: str):
        """Set the path for the FODF file after validation."""
        if self._isValidPath(path):
            self.trkPath = path
            print(f"Trk path set to: {self.trkPath}")
        else:
            raise FileNotFoundError(f"Invalid trk path: {path}")

    def set_stepSize(self, step: float):
        """Set the step size with validation."""
        step = float(step)
        if isinstance(step, (int, float)) and step > 0:
            self.stepSize = float(step)
            print(f"Step size set to: {self.stepSize}")

    def set_algo(self, index: int):
        """Set the tracking algorithm with validation."""
        algos = ["deterministic", "probalis", "eudx"]
        self.algo = algos[index]
        slicer.util.showStatusMessage(f"Selected: {self.algo}", 2000)
        print(f"Algorithm set to: {self.algo}")

    def generateTrk(self):
        """Trigger the tractography pipeline on button click."""
        print("GenerateTrk button clicked.")

        seeding_mask_file = self.approxMaskPath
        fodf_file = self.fodfPath
        output_file = self.trkPath

        # print(f"Seeding Mask Path: {seeding_mask_file}")
        # print(f"FODF File Path: {fodf_file}")
        # print(f"Output File Path: {output_file}")

        step_size = self.stepSize
        algotype = self.algo

        processor = SyncthingTractographyProcessor(
            seeding_mask_file=os.path.basename(seeding_mask_file),
            fodf_file=os.path.basename(fodf_file),
            output_file=os.path.basename(output_file),
            step_size=step_size,
            algotype=algotype
        )

        processor.run_pipeline()
    
    def saveStreamlinesVTK(self, streamlines, pStreamlines):
        """
        Save streamlines to VTK file format
        
        Args:
            streamlines: List of streamline coordinates
            pStreamlines: Output path for VTK file
        """
        polydata = vtk.vtkPolyData()
        lines = vtk.vtkCellArray()
        points = vtk.vtkPoints()
        ptCtr = 0
        
        for i, streamline in enumerate(streamlines):
            if (i % 10000) == 0:
                print(f"{i}/{len(streamlines)}")
                
            line = vtk.vtkLine()
            line.GetPointIds().SetNumberOfIds(len(streamline))
            
            for j, point in enumerate(streamline):
                points.InsertNextPoint(point)
                line.GetPointIds().SetId(j, ptCtr)
                ptCtr += 1
                
            lines.InsertNextCell(line)
        
        polydata.SetLines(lines)
        polydata.SetPoints(points)
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(pStreamlines)
        writer.SetInputData(polydata)
        writer.Write()
        print(f"Wrote streamlines to {writer.GetFileName()}")

    def visualizeTrk(self):
        """Visualize tractography data in 3D Slicer."""
        if self.output_trk_path is None:
            print("Generate Trk First...")
            return
            
        tractogram = load_tractogram(
            self.output_trk_path, 
            reference='same', 
            bbox_valid_check=False, 
            to_space=Space.RASMM
        )
        print("Tractography Visualization...")
        
        if self.trkPath:
            output_vtk_path = os.path.join(self.trkPath, DEFAULT_VTK_FILE_NAME)
        else:
            raise ValueError("trkPath is not set")
            
        # Call the function without underscore
        self.saveStreamlinesVTK(tractogram.streamlines, output_vtk_path)
        
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(output_vtk_path)
        reader.Update()
        polydata = reader.GetOutput()
        
        streamline_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        streamline_node.SetAndObservePolyData(polydata)
        display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        streamline_node.SetAndObserveDisplayNodeID(display_node.GetID())
        
        # Set visualization properties
        display_node.SetColor(0, 1, 0)  # Green color
        display_node.SetOpacity(1.0)
        
        # Update view
        slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveVolumeID(streamline_node.GetID())
        slicer.app.applicationLogic().PropagateVolumeSelection()
        slicer.app.layoutManager().resetThreeDViews()
        
        print("Done Visualization")
        print(f"VTK Files Generated Successfully (location: {output_vtk_path})\nVisualization Complete\n")
