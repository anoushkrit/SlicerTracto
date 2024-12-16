import logging
import os
from typing import Annotated, Optional
import sys
module_path = os.path.join(os.path.dirname(__file__), "Modules")
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.join(os.path.dirname(__file__), "scilpy")
if module_path not in sys.path:
    sys.path.append(module_path)
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
import qt

from generateFodfModule import GenerateFODF
from tractographyModule import Tractography
from metricAnalysisModule import MetricAnalysis
from segmentationModule import Segmentation
from Middleware.middleware import Middleware
from SSH_Connection.ssh_connection import SSHConnection

#
# DMRI_Tractography
#


class DMRI_Tractography(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("DMRI_Tractography")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "DMRI_Tractography")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#DMRI_Tractography">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
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

    # DMRI_Tractography1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="DMRI_Tractography",
        sampleName="DMRI_Tractography1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "DMRI_Tractography1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="DMRI_Tractography1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="DMRI_Tractography1",
    )

    # DMRI_Tractography2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="DMRI_Tractography",
        sampleName="DMRI_Tractography2",
        thumbnailFileName=os.path.join(iconsPath, "DMRI_Tractography2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="DMRI_Tractography2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="DMRI_Tractography2",
    )


#
# DMRI_TractographyWidget
#


class DMRI_TractographyWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
       
        
        self._generateFodfParams : GenerateFODF = GenerateFODF()
        self._tractographyParams : Tractography = Tractography()
        self._metricAnalysis : MetricAnalysis = MetricAnalysis()
        self._segmentationParams : Segmentation = Segmentation()
        self._middleware : Middleware =  Middleware()
        self._sshConnection: SSHConnection = SSHConnection()

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/DMRI_Tractography.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.sshConnectionCheckBox.connect(self._middleware.setToggle)
        self.ui.sshConnectionButton.connect(self._sshConnection.connect)

        self.setup_generate_fodf_connections()
        self.setup_tractography_connections()
        self.setup_metric_analysis_connections()
        self.setup_segementation_connections()
    
    def setup_generate_fodf_connections(self):

        self._middleware.registerButton(self.ui.generateFodfButton, self._generateFodfParams.generateFodf, self._generateFodfParams.generateFodf)
        self._middleware.registerButton(self.ui.visualizeFodfButton, self._generateFodfParams.visualizeFodf, self._generateFodfParams.visualizeFodf)
        self._middleware.registerButton(self.ui.whiteMaskBiftiPath, self._generateFodfParams.setWhiteMaskBiftiPath, self._generateFodfParams.setWhiteMaskBiftiPath)
        self._middleware.registerButton(self.ui.diffusionNiftiPath, self._generateFodfParams.setDiffusionNiftiPath, self._generateFodfParams.setDiffusionNiftiPath)
        self._middleware.registerButton(self.ui.bvalsPath, self._generateFodfParams.setBvalsPath, self._generateFodfParams.setBvalsPath)
        self._middleware.registerButton(self.ui.bvecsPath, self._generateFodfParams.setBvecsPath, self._generateFodfParams.setBvecsPath)
        self._middleware.registerButton(self.ui.fodfPath,self._generateFodfParams.setFodfPath, self._generateFodfParams.setFodfPath)

        # Generate FODF Module Connections
        # Buttons
        self.ui.generateFodfButton.connect("clicked(bool)", lambda: self._middleware.handleButtonPress(self.ui.generateFodfButton))
        self.ui.visualizeFodfButton.connect("clicked(bool)", lambda: self._middleware.handleButtonPress(self.ui.visualizeFodfButton))
        # Paths
        self.ui.whiteMaskBiftiPath.connect('currentPathChanged(QString)', lambda: self._middleware.handleButtonPress(self.ui.whiteMaskBiftiPath))
        self.ui.diffusionNiftiPath.connect('currentPathChanged(QString)', lambda: self._middleware.handleButtonPress(self.ui.diffusionNiftiPath))
        self.ui.bvalsPath.connect('currentPathChanged(QString)', lambda: self._middleware.handleButtonPress(self.ui.bvalsPath))
        self.ui.bvecsPath.connect('currentPathChanged(QString)', lambda: self._middleware.handleButtonPress(self.ui.bvecsPath))
        self.ui.fodfPath.connect('currentPathChanged(QString)', lambda: self._middleware.handleButtonPress(self.ui.fodfPath))
        # Output Text Box
        self._generateFodfParams.outputText = self.ui.outputTextGeneratedFodf

    def setup_tractography_connections(self):

        self._middleware.registerButton(self.ui.generateTrkButton, self._tractographyParams.generateTrk, self._tractographyParams.generateTrk)
        self._middleware.registerButton(self.ui.visualizeTrkButton, self._tractographyParams.visualizeTrk, self._tractographyParams.visualizeTrk)
        self._middleware.registerButton(self.ui.approxMaskPath, self._tractographyParams.set_approxMaskPath, self._tractographyParams.set_approxMaskPath)
        self._middleware.registerButton(self.ui.fodfTractographyPath, self._tractographyParams.set_fodfPath, self._tractographyParams.set_fodfPath)
        self._middleware.registerButton(self.ui.approxMaskPath, self._tractographyParams.set_approxMaskPath, self._tractographyParams.set_approxMaskPath)
        self._middleware.registerButton(self.ui.trkPath, self._tractographyParams.set_trkPath, self._tractographyParams.set_trkPath)
        self._middleware.registerButton(self.ui.stepSize.textChanged, self._tractographyParams.set_stepSize, self._tractographyParams.set_stepSize)
        self._middleware.registerButton(self.ui.algo.currentIndexChanged, self._tractographyParams.set_algo, self._tractographyParams.set_algo)

        # Tractography Module Connections
        # Buttons
        self.ui.generateTrkButton.connect("clicked(bool)", lambda: self._middleware.handleButtonPress(self.ui.generateTrkButton))
        self.ui.visualizeTrkButton.connect("clicked(bool)", lambda: self._middleware.handleButtonPress(self.ui.visualizeTrkButton))
        # Paths
        self.ui.approxMaskPath.connect('currentPathChanged(QString)', lambda: self._middleware.handleButtonPress(self.ui.approxMaskPath))
        self.ui.fodfTractographyPath.connect('currentPathChanged(QString)', lambda: self._middleware.handleButtonPress(self.ui.fodfTractographyPath))
        self.ui.approxMaskPath.connect('currentPathChanged(QString)', lambda: self._middleware.handleButtonPress(self.ui.approxMaskPath))
        self.ui.trkPath.connect('currentPathChanged(QString)', lambda: self._middleware.handleButtonPress(self.ui.trkPath))
        # Text
        validator = qt.QDoubleValidator()
        validator.setNotation(qt.QDoubleValidator.StandardNotation)
        validator.setDecimals(4)  # Allow up to 4 decimal places
        validator.setRange(-100.0, 100.0) 
        self.ui.stepSize.setValidator(validator)
        self.ui.stepSize.textChanged.connect(lambda: self._middleware.handleButtonPress(self.ui.tepSize.textChanged))
        #Combo Box
        self.ui.algo.currentIndexChanged.connect(lambda: self._middleware.handleButtonPress(self.ui.algo.currentIndexChanged))
        #Output Box
        self._tractographyParams.outputText = self.ui.outputTextTractography

    def setup_metric_analysis_connections(self):
        # Metrix Analysis Module Connections
        # Buttons
        self.ui.generateResultsButton.connect("clicked(bool)", self._metricAnalysis.generateMetrics)
        # Paths
        self.ui.predictedTrkPath.connect('currentPathChanged(QString)', self._metricAnalysis.setPredictedTrkPath)
        self.ui.groundTruthTrkPath.connect('currentPathChanged(QString)', self._metricAnalysis.setGroundTruthTrkPath)
        # Initializing Outputs Text
        self.ui.diceScore.setText(f'Dice Score: -')
        self.ui.overlapScore.setText(f'Overlap: -')
        self.ui.overreachScore.setText(f'Overreach: -')
        # passing UI parameter
        self._metricAnalysis.ui = self.ui

    def setup_segementation_connections(self):
        # Segmenation Module Connections
        # Buttons
        self.ui.segmentationButton_Segmentation.connect("clicked(bool)", self._segmentationParams.segmentTrk)
        self.ui.visualizeTrks_Segmentation.connect("clicked(bool)", self._segmentationParams.visualizeSegmentation)
        # Path
        self.ui.trkPath_Segmentation.connect('currentPathChanged(QString)', self._segmentationParams.set_trkPath)
        self.ui.segmentedTrkFolderPath_Segmentation.connect('currentPathChanged(QString)', self._segmentationParams.set_segmentedTrkFolderPath)
        # Output Text Box
        self._segmentationParams.outputText = self.ui.outputText_Segmenatation
        # Slider 
        self.ui.threasholdInput.valueChanged.connect(self._segmentationParams.set_threshold)

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        pass

    def exit(self) -> None:
        pass

    def onSceneStartClose(self, caller, event) -> None:
        pass

    def onSceneEndClose(self, caller, event) -> None:
        pass

