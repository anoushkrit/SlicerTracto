import logging
import os
from typing import Annotated, Optional
from __main__ import vtk, qt, ctk, slicer

import vtk
# from vtk.util import numpy_support

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)


import logging

from dipy.core.sphere import HemiSphere
from dipy.data import get_sphere
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter)
from dipy.direction.peaks import PeaksAndMetrics
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamlinespeed import length
from dipy.tracking import utils as track_utils
import nibabel as nib
import numpy as np
import warnings

from dipy.direction.peaks import peak_directions
from dipy.reconst.shm import sph_harm_lookup
# from scilpy.reconst.utils import (find_order_from_nb_coeff,
#                                   get_b_matrix, get_maximas)

from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.io.stateful_tractogram import StatefulTractogram, Space

from slicer import vtkMRMLScalarVolumeNode


#
# Tractography
#


class Tractography(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Tractography")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Mahir (IIT Mandi)"]  
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#Tractography">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Mahir Jain and is funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # Tractography1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="Tractography",
        sampleName="Tractography1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "Tractography1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="Tractography1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="Tractography1",
    )

    # Tractography2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="Tractography",
        sampleName="Tractography2",
        thumbnailFileName=os.path.join(iconsPath, "Tractography2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="Tractography2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="Tractography2",
    )


#
# TractographyParameterNode
#


@parameterNodeWrapper
class TractographyParameterNode:

    inputVolume1: vtkMRMLScalarVolumeNode
    inputVolume2: vtkMRMLScalarVolumeNode

    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode

#
# TractographyWidget
#


class TractographyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Collapsible button
        self.importCollapsibleButton = ctk.ctkCollapsibleButton()
        self.importCollapsibleButton.text = "Import FODF NIfTI"
        self.layout.addWidget(self.importCollapsibleButton)
        
        # Layout within the dummy collapsible button
        self.formLayout = qt.QFormLayout(self.importCollapsibleButton)

        # Add input directory selector for NIfTI file
        self.niftiSelector = ctk.ctkPathLineEdit()
        self.niftiSelector.filters = ctk.ctkPathLineEdit.Files
        self.formLayout.addRow("Input NIfTI File:", self.niftiSelector)

        # Add output node selector
        self.outputNodeSelector = slicer.qMRMLNodeComboBox()
        self.outputNodeSelector.nodeTypes = ['vtkMRMLMultiVolumeNode']
        self.outputNodeSelector.setMRMLScene(slicer.mrmlScene)
        self.outputNodeSelector.addEnabled = 1
        self.formLayout.addRow("Output Node:", self.outputNodeSelector)


        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Tractography.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.invertedInputSelector.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = TractographyLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume1:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume1 = firstVolumeNode
                
        if not self._parameterNode.inputVolume2:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume2 = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[TractographyParameterNode]) -> None:

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume1 and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False
        
    # def onImportButtonClicked(self):
    #     niftiFilePath = self.niftiSelector.currentPath
    #     mvNode = self.outputNodeSelector.currentNode()

    #     if not mvNode or not os.path.exists(niftiFilePath):
    #         slicer.util.errorDisplay("Invalid input NIfTI file or output node.")
    #         return

    #     self.read4DFODFNIfTI(mvNode, niftiFilePath)

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""

        mvNode = self.outputNodeSelector.currentNode()
        niftiFilePath = self.niftiSelector.currentPath
        
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedInputSelector.currentNode(), mvNode, "trk_1061_mrm_detT.trk", niftiFilePath,
                            self.ui.invertOutputCheckBox.checked)

#
# TractographyLogic
#

class TractographyLogic(ScriptedLoadableModuleLogic):
    

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return TractographyParameterNode(super().getParameterNode())
    
    def find_order_from_nb_coeff(self, data):
        if isinstance(data, np.ndarray):
            shape = data.shape
        else:
            shape = data
        return int((-3 + np.sqrt(1 + 8 * shape[-1])) / 2)
    
    def readFODFNIfTI(self, mvNode, niftiFilePath):
        """Read a 4D NIfTI file containing spherical harmonics as multivolume."""

        # Use VTK's NIfTI reader to load the 4D NIfTI file
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(niftiFilePath)
        reader.SetTimeAsVector(True)  # Use the 4th dimension
        reader.Update()
        
        nFrames = reader.GetTimeDimension()

        if nFrames < 1:
            slicer.util.errorDisplay("Invalid 4D NIfTI file: no frames detected.")
            return

        print(f"Successfully read {nFrames} spherical harmonics frames from the FODF NIfTI file.")

        # Display message with the number of frames (SH coefficients)
        mvNode.SetName(f"{nFrames} frames NIfTI MultiVolume")
        slicer.util.infoDisplay(f"Successfully imported {nFrames} frames from the FODF NIfTI file.")

        # Process and store the FODF data
        mvImage = reader.GetOutputDataObject(0)

        # Create and configure the MultiVolume display node
        mvDisplayNode = slicer.mrmlScene.CreateNodeByClass('vtkMRMLMultiVolumeDisplayNode')
        mvDisplayNode.SetScene(slicer.mrmlScene)
        slicer.mrmlScene.AddNode(mvDisplayNode)
        mvDisplayNode.SetDefaultColorMap()

        # Set the image data and display node for the multivolume node
        mvNode.SetAndObserveImageData(mvImage)
        mvNode.SetAndObserveDisplayNodeID(mvDisplayNode.GetID())
        mvNode.SetNumberOfFrames(nFrames)

        print(f"Output node configured with {nFrames} frames.")

        # Optionally, you can add the logic to handle frame labels or SH-specific attributes

        # Return the number of frames (spherical harmonics coefficients)
        return nFrames

    

    def _honor_authorsnames_sh_basis(self, sh_basis_type):
        sh_basis = sh_basis_type
        if sh_basis_type == 'fibernav':
            sh_basis = 'descoteaux07'
            warnings.warn("'fibernav' sph basis name is deprecated and will be "
                        "discontinued in favor of 'descoteaux07'.",
                        DeprecationWarning)
        elif sh_basis_type == 'mrtrix':
            sh_basis = 'tournier07'
            warnings.warn("'mrtrix' sph basis name is deprecated and will be "
                        "discontinued in favor of 'tournier07'.",
                        DeprecationWarning)
        return sh_basis

    
    def get_b_matrix(self, order, sphere, sh_basis_type, return_all=False):
        sh_basis = self._honor_authorsnames_sh_basis(sh_basis_type)
        sph_harm_basis = sph_harm_lookup.get(sh_basis)
        if sph_harm_basis is None:
            raise ValueError("Invalid basis name.")
        b_matrix, m, n = sph_harm_basis(order, sphere.theta, sphere.phi)
        if return_all:
            return b_matrix, m, n
        return b_matrix


    def get_maximas(self, data, sphere, b_matrix, threshold, absolute_threshold,
                    min_separation_angle=25):
        spherical_func = np.dot(data, b_matrix.T)
        spherical_func[np.nonzero(spherical_func < absolute_threshold)] = 0.
        return peak_directions(
            spherical_func, sphere, threshold, min_separation_angle)

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                inputVolume2: vtkMRMLScalarVolumeNode,
                mvNode: vtkMRMLScalarVolumeNode,
                outputTrkFilePath: str,             
                niftiFilePath: str,             
                step_size: float = 0.2,
                algorithm: str = 'det',
                npv: int = 7,
                verbose: bool = False,
                ) -> None:
        

        print("The inputVolume is ", inputVolume)
        print("The inputVolume is, ", inputVolume2)
        print("The niftiFilePath is ", niftiFilePath)
        # print("The outputTrkFilePath is, ", outputTrkFilePath)

        if not inputVolume or not inputVolume2:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Convert Slicer volumes to numpy arrays
        frames = self.readFODFNIfTI(mvNode, niftiFilePath)
        fodf_data = slicer.util.arrayFromVolume(inputVolume).astype(np.float32)
        
        # vtk_image_data = inputVolume.GetImageData()
        # dimensions = vtk_image_data.GetDimensions()
        # print("Dimensions of the input volume:", dimensions)  # Check the dimensions
        # vtk_array = vtk_image_data.GetPointData().GetScalars()
        # numpy_data = numpy_support.vtk_to_numpy(vtk_array)
        # x, y, z = dimensions[:3]
        # sh_coeff_count = numpy_data.size // (x * y * z)
        # numpy_data_reshaped = numpy_data.reshape((x, y, z, sh_coeff_count))
        # print("Reshaped numpy data to 4D:", numpy_data_reshaped.shape)

        seed_mask_data = slicer.util.arrayFromVolume(inputVolume2).astype(np.float32)

        fodf_4d = np.reshape(fodf_data, (fodf_data.shape[2], fodf_data.shape[1], fodf_data.shape[0], frames))

        print("Shape of fodf_data:", fodf_4d.shape)
        print("the dimensions of fodf : ", frames)
      
        affine = np.eye(4)

        spacing = inputVolume.GetSpacing()
        if not np.allclose(np.mean(spacing), spacing[0], atol=1e-03):
            raise ValueError('ODF SH file is not isotropic. Tracking cannot be run robustly.')

        voxel_size = spacing[0]
        vox_step_size = step_size / voxel_size

        seeds = track_utils.random_seeds_from_mask(
            seed_mask_data,
            affine,
            seeds_count=npv,
            seed_count_per_voxel=True,
            random_seed=None)

        mask_data = seed_mask_data > 0
        stopping_criterion = BinaryStoppingCriterion(mask_data)

        sphere = HemiSphere.from_sphere(get_sphere('symmetric724'))
        theta = 45.0
        sh_basis = 'tournier07'  
        sf_threshold = 0.1  

        non_zeros_count = np.count_nonzero(np.sum(fodf_4d, axis=-1))
        non_first_val_count = np.count_nonzero(np.argmax(fodf_4d, axis=-1))

        if algorithm in ['det', 'prob']:
            if non_first_val_count / non_zeros_count > 0.5:
                logging.warning('Input detected as peaks. Input should be fodf for det/prob; verify input just in case.')
            if algorithm == 'det':
                dg_class = DeterministicMaximumDirectionGetter
            else:
                dg_class = ProbabilisticDirectionGetter
            direction_getter = dg_class.from_shcoeff(
                shcoeff=fodf_4d,
                max_angle=theta,
                sphere=sphere,
                basis_type=sh_basis,
                relative_peak_threshold=sf_threshold)
        elif algorithm == 'eudx':
            odf_shape_3d = fodf_4d.shape[:-1]
            dg = PeaksAndMetrics()
            dg.sphere = sphere
            dg.ang_thr = theta
            dg.qa_thr = sf_threshold

            if non_first_val_count / non_zeros_count > 0.5:
                logging.info('Input detected as peaks.')
                nb_peaks = fodf_4d.shape[-1] // 3
                slices = np.arange(0, nb_peaks * 3 + 1, 3)
                peak_values = np.zeros(odf_shape_3d + (nb_peaks,))
                peak_indices = np.zeros(odf_shape_3d + (nb_peaks,))

                for idx in np.argwhere(np.sum(fodf_4d, axis=-1)):
                    idx = tuple(idx)
                    for i in range(nb_peaks):
                        vec = fodf_4d[idx][slices[i]:slices[i+1]]
                        peak_values[idx][i] = np.linalg.norm(vec, axis=-1)
                        peak_indices[idx][i] = sphere.find_closest(vec)

                dg.peak_dirs = fodf_4d
            else:
                logging.info('Input detected as fodf.')
                npeaks = 5
                peak_dirs = np.zeros(odf_shape_3d + (npeaks, 3))
                peak_values = np.zeros(odf_shape_3d + (npeaks,))
                peak_indices = np.full(odf_shape_3d + (npeaks,), -1, dtype='int')
                b_matrix = self.get_b_matrix(self,
                    self.find_order_from_nb_coeff(fodf_4d), sphere, sh_basis)

                for idx in np.argwhere(np.sum(fodf_4d, axis=-1)):
                    idx = tuple(idx)
                    directions, values, indices = self.get_maximas(fodf_4d[idx],
                                                            sphere, b_matrix,
                                                            sf_threshold, 0)
                    if values.shape[0] != 0:
                        n = min(npeaks, values.shape[0])
                        peak_dirs[idx][:n] = directions[:n]
                        peak_values[idx][:n] = values[:n]
                        peak_indices[idx][:n] = indices[:n]

                dg.peak_dirs = peak_dirs

            dg.peak_values = peak_values
            dg.peak_indices = peak_indices

            direction_getter = dg

        else:
            raise ValueError(f"Unknown algorithm '{algorithm}'")

        max_length = 200.0  
        max_steps = int(max_length / step_size) + 1

        streamlines_generator = LocalTracking(
            direction_getter,
            stopping_criterion,
            seeds,
            affine,
            step_size=vox_step_size,
            max_cross=1,
            maxlen=max_steps,
            fixedstep=True,
            return_all=True,
            random_seed=None,
            save_seeds=False)

        min_length = 10.0
        scaled_min_length = min_length / voxel_size
        scaled_max_length = max_length / voxel_size

        filtered_streamlines = (
            s for s in streamlines_generator
            if scaled_min_length <= length(s) <= scaled_max_length)

        tractogram = StatefulTractogram(
            streamlines=filtered_streamlines,
            data_per_streamline={},
            data_per_point={},
            affine_to_rasmm=affine,
            space=Space.RASMM)

        nib.streamlines.save(tractogram, outputTrkFilePath)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")


#
# TractographyTest
#

