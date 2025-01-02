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

import importlib
# import Modules.tractographyModule as tractographyModule
# import Modules.tractographyModule_2 as tractographyModule_2

import Modules.generateFodfModule as generateFodfModule
import Modules.generateFodfModule_2 as generateFodfModule_2

import Modules.tractographyModule as tractographyModule
import Modules.tractographyModule_2 as tractographyModule_2

from Modules.metricAnalysisModule import MetricAnalysis
# from Modules.metricAnalysisModule_2 import MetricAnalysis as MetricAnalysis_2

import Modules.segmentationModule as segmentationModule
import Modules.segmentationModule_2 as segmentationModule_2 


# from generateFodfModule import GenerateFODF
# from tractographyModule_2 import Tractography
# from metricAnalysisModule import MetricAnalysis
# from segmentationModule_2 import Segmentation

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
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "SlicerTracto")]
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
# DMRI_TractographyParameterNode
#


@parameterNodeWrapper
class DMRI_TractographyParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    imageThreshold: Annotated[float, WithinRange(-100, 500)] = 100
    invertThreshold: bool = False
    thresholdedVolume: vtkMRMLScalarVolumeNode
    invertedVolume: vtkMRMLScalarVolumeNode


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
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        
        self.isSSH = False 
        
        self._generateFodfParams : generateFodfModule.GenerateFODF = generateFodfModule.GenerateFODF()
        self._tractographyParams : tractographyModule.Tractography = tractographyModule.Tractography()
        self._metricAnalysis : MetricAnalysis = MetricAnalysis()
        self._segmentationParams : segmentationModule.Segmentation = segmentationModule.Segmentation()

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/DMRI_Tractography.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        uiWidget.setMRMLScene(slicer.mrmlScene)
        self.logic = DMRI_TractographyLogic()
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # adding a UI button to switch from local to SSH
        self.ui.toggleImportButton.clicked.connect(self.toggleImport)

        # Generate FODF Module Connections
        self._connect_generate_fodf_module()

        # Tractography Module Connections
        self._connect_tractography_module()

        # Metric Analysis Module Connections
        self._connect_metric_analysis_module()

        # Segmentation Module Connections
        self._connect_segmentation_module()

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def _connect_generate_fodf_module(self):
        # Buttons
        self.ui.generateFodfButton.connect("clicked(bool)", self._generateFodfParams.generateFodf)
        self.ui.visualizeFodfButton.connect("clicked(bool)", self._generateFodfParams.visualizeFodf)
        # Paths
        self.ui.whiteMaskBiftiPath.connect('currentPathChanged(QString)', self._generateFodfParams.setWhiteMaskBiftiPath)
        self.ui.diffusionNiftiPath.connect('currentPathChanged(QString)', self._generateFodfParams.setDiffusionNiftiPath)
        self.ui.bvalsPath.connect('currentPathChanged(QString)', self._generateFodfParams.setBvalsPath)
        self.ui.bvecsPath.connect('currentPathChanged(QString)', self._generateFodfParams.setBvecsPath)
        self.ui.fodfPath.connect('currentPathChanged(QString)', self._generateFodfParams.setFodfPath)
        # Output Text Box
        self._generateFodfParams.outputText = self.ui.outputTextGeneratedFodf

    def _connect_tractography_module(self):
        # Buttons
        self.ui.generateTrkButton.connect("clicked(bool)", self._tractographyParams.generateTrk)
        self.ui.visualizeTrkButton.connect("clicked(bool)", self._tractographyParams.visualizeTrk)
        # Paths
        self.ui.approxMaskPath.connect('currentPathChanged(QString)', self._tractographyParams.set_approxMaskPath)
        self.ui.fodfTractographyPath.connect('currentPathChanged(QString)', self._tractographyParams.set_fodfPath)
        self.ui.trkPath.connect('currentPathChanged(QString)', self._tractographyParams.set_trkPath)
        # Text
        validator = qt.QDoubleValidator()
        validator.setNotation(qt.QDoubleValidator.StandardNotation)
        validator.setDecimals(4)  # Allow up to 4 decimal places
        validator.setRange(-100.0, 100.0)
        self.ui.stepSize.setValidator(validator)
        self.ui.stepSize.textChanged.connect(self._tractographyParams.set_stepSize)
        # Combo Box
        self.ui.algo.currentIndexChanged.connect(self._tractographyParams.set_algo)
        # Output Box
        self._tractographyParams.outputText = self.ui.outputTextTractography

    def _connect_metric_analysis_module(self):
        # Buttons
        self.ui.generateResultsButton.connect("clicked(bool)", self._metricAnalysis.generateMetrics)
        # Paths
        self.ui.predictedTrkPath.connect('currentPathChanged(QString)', self._metricAnalysis.setPredictedTrkPath)
        self.ui.groundTruthTrkPath.connect('currentPathChanged(QString)', self._metricAnalysis.setGroundTruthTrkPath)
        # Initializing Outputs Text
        self.ui.diceScore.setText(f'Dice Score: -')
        self.ui.overlapScore.setText(f'Overlap: -')
        self.ui.overreachScore.setText(f'Overreach: -')
        # Passing UI parameter
        self._metricAnalysis.ui = self.ui

    def _connect_segmentation_module(self):
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

    def toggleImport(self):
        # Toggle between SSH and Local
        self.isSSH = not self.isSSH
        self.ui.toggleImportButton.setText("Switch to Local" if self.isSSH else "Switch to SSH")

        # Reload GenerateFODF
        generateFodfModuleReloaded = importlib.reload(generateFodfModule_2 if self.isSSH else generateFodfModule)
        self._generateFodfParams = generateFodfModuleReloaded.GenerateFODF()
        self._connect_generate_fodf_module()

        # Reload Tractography
        tractographyModuleReloaded = importlib.reload(tractographyModule_2 if self.isSSH else tractographyModule)
        self._tractographyParams = tractographyModuleReloaded.Tractography()
        self._connect_tractography_module()

        # Reload Segmentation
        segmentationModuleReloaded = importlib.reload(segmentationModule_2 if self.isSSH else segmentationModule)
        self._segmentationParams = segmentationModuleReloaded.Segmentation()
        self._connect_segmentation_module()


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
        # if not self._parameterNode.inputVolume:
        #     firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #     if firstVolumeNode:
        #         self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[DMRI_TractographyParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            # self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            # self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            # self._checkCanApply()

    # def _checkCanApply(self, caller=None, event=None) -> None:
    #     pass
    #     # if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
    #     #     self.ui.applyButton.toolTip = _("Compute output volume")
    #     #     self.ui.applyButton.enabled = True
    #     # else:
    #     #     self.ui.applyButton.toolTip = _("Select input and output volume nodes")
    #     #     self.ui.applyButton.enabled = False

    # def onApplyButton(self) -> None:
    #     """Run processing when user clicks "Apply" button."""
    #     with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
    #         # Compute output
    #         self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
    #                            self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

    #         # Compute inverted output (if needed)
    #         if self.ui.invertedOutputSelector.currentNode():
    #             # If additional output volume is selected then result with inverted threshold is written there
    #             self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
    #                                self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)


#
# DMRI_TractographyLogic
#


class DMRI_TractographyLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return DMRI_TractographyParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# DMRI_TractographyTest
#


class DMRI_TractographyTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_DMRI_Tractography1()

    def test_DMRI_Tractography1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("DMRI_Tractography1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = DMRI_TractographyLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
