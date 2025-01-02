
from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles
from dipy.io.streamline import save_trk
from dipy.tracking.streamline import transform_streamlines
from dipy.io.streamline import load_tractogram
from dipy.io.streamline import save_tractogram
import vtk
import os
import slicer
import random

DEFAULT_SEGMENTED_TRK_FILE_NAME_PREFIX = 'segmentedTrk'
file_path = os.path.abspath(__file__)
DEFAULT_DIR = os.path.dirname(file_path)

class Segmentation:
    def _init_(self):
        self.trkPath: str = None
        self.segmentedTrkFolderPath : str = DEFAULT_DIR
        self.outputText = None
        self.clusters = None
        self.threshold : float = 10.0
    
    @staticmethod
    def _isValidPath(path: str) -> bool:
        """Validate if the provided path exists and is a file."""
        return os.path.isfile(path) or os.path.isdir(path)

    def set_trkPath(self, path: str):
        """Set the path for the FODF file after validation."""
        if self._isValidPath(path):
            self.trkPath = path
            print(f"Trk path set to: {self.trkPath}")
        else:
            raise FileNotFoundError(f"Invalid trk path: {path}")

    def set_segmentedTrkFolderPath(self, path: str):
        """Set the path for the FODF file after validation."""
        if self._isValidPath(path):
            self.segmentedTrkFolderPath = path
            print(f"Trks Folder path set to: {self.segmentedTrkFolderPath}")
        else:
            raise FileNotFoundError(f"Trks Folder path: {path}")
    
    def set_threshold(self, threshold):
        self.threshold = float(threshold)
        print(f"Threashold value changed to {threshold}")

    
    def segmentTrk(self):
        print("Segmenting Trk...")
        tractogram = load_tractogram(self.trkPath, reference="same")
        streamlines = tractogram.streamlines
        # Define the threshold for clustering (in mm)
          # Adjust based on desired clustering sensitivity

        # Initialize QuickBundles
        qb = QuickBundles(threshold=self.threshold)

        # Perform clustering
        self.clusters = qb.cluster(streamlines)

        self.outputText.append(f"Segmentation Completed... \n No. of Clusters = {len(self.clusters)} \n")

        for i, cluster in enumerate(self.clusters):
            self.outputText.append(f"Cluster {i}: {len(cluster)} streamlines \n")
            # Extract streamlines for the cluster
            cluster_streamlines = [streamlines[idx] for idx in cluster.indices]
            
            # Save the cluster to a new .trk file
            trkFileName = DEFAULT_SEGMENTED_TRK_FILE_NAME_PREFIX + f"-{i}.trk"
            vtkFileName = DEFAULT_SEGMENTED_TRK_FILE_NAME_PREFIX + f"-{i}.vtk"

            trkFilePath = os.path.join(self.segmentedTrkFolderPath, trkFileName)
            vtkFilePath = os.path.join(self.segmentedTrkFolderPath, vtkFileName)

            save_tractogram(tractogram, trkFilePath)
            tractogram = load_tractogram(trkFilePath, reference="same")
            self._saveStreamlinesVTK(tractogram.streamlines, vtkFilePath )
            self.outputText.append(f'Cluster {i} saved in file {trkFileName} \n\n')

    def visualizeSegmentation(self):
        """Visualize clustering results using VTK in Slicer3D."""
        if not self.clusters:
            print("No self.clusters to visualize. Please run `segment` first.")
            return

        try:
            colors = self._generateColors(len(self.clusters))

            for idx, cluster in enumerate(self.clusters):
                vtk_polydata = self._convertToVTK(cluster)

                # Create a new model node
                modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", f"Cluster {idx + 1}")
                modelNode.SetAndObservePolyData(vtk_polydata)

                # Add a display node explicitly
                displayNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
                modelNode.SetAndObserveDisplayNodeID(displayNode.GetID())

                # Set color and scalar visibility
                displayNode.SetColor(colors[idx])
                displayNode.SetScalarVisibility(True)
                displayNode.SetActiveScalarName("ClusterIndex")
                displayNode.SetScalarRange(0, len(cluster))

                displayNode.SetVisibility(True)
                self.outputText.append(f"\n Ouput Clusters Visualized with color {colors[idx]}")

            print("Clusters visualized successfully in Slicer3D.")

        except Exception as e:
            print(f"An error occurred during visualization: {e}")


    def _convertToVTK(self, cluster):
        """Convert a cluster of streamlines to VTK polydata with per-cluster coloring."""
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        scalars = vtk.vtkFloatArray()
        scalars.SetName("ClusterIndex")

        for streamline in cluster:
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(len(streamline))

            for i, point in enumerate(streamline):
                pointId = points.InsertNextPoint(point)
                line.GetPointIds().SetId(i, pointId)
                scalars.InsertNextValue(i)  # Assign scalar value (index)

            lines.InsertNextCell(line)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        polydata.GetPointData().SetScalars(scalars)  # Add scalars for coloring

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
    
    def _saveStreamlinesVTK(self, streamlines, pStreamlines):
        
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
            


        
