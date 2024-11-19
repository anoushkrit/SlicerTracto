import os
import sys
import logging

# Get the path to the scilpy folder and add to sys paths
sibling_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scilpy'))
if sibling_folder_path not in sys.path:
    sys.path.append(sibling_folder_path)

from scilpy.reconst.utils import (find_order_from_nb_coeff,
                                    get_maximas)
from scilpy.io.image import get_data_as_mask
from scilpy.tracking.utils import get_theta

from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking.streamlinespeed import length
from dipy.tracking import utils as track_utils
from dipy.io.streamline import save_trk
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
from dipy.io.utils import is_header_compatible
from dipy.io.streamline import load_tractogram
from dipy.io.image import load_nifti

from dipy.io.utils import (get_reference_info,
                           create_tractogram_header)
from dipy.reconst.shm import sph_harm_lookup
from dipy.direction.peaks import PeaksAndMetrics
from dipy.direction import (DeterministicMaximumDirectionGetter,
                            ProbabilisticDirectionGetter) 
from dipy.core.sphere import HemiSphere
from dipy.data import get_sphere
from vtk import vtkPolyDataReader
import numpy as np
import warnings
import nibabel as nib
from nibabel.streamlines.tractogram import LazyTractogram
from scripts.scil_frf_ssst import main as scil_frf_ssst_main
import slicer.util
import slicer
import vtk


# Constansts 
NPV = None
NT = None
MIN_LENGTH : float = 10.0
MAX_LENGTH : float = 200.0
SPHERE : str = 'symmetric724'
THETA : float = 45.0
SH_BASIS : str = 'tournier07'  
SF_THRESHOLD : float = 0.1
SEED = None
SAVE_SEEDS = False
FILETYPE = nib.streamlines.TrkFile
COMPRESS : float = 0.0
SEEDING_MASK_FILE_PATH: str = "C:/Users/HP/Documents/MTP/Slicer-Task/Final Modules/Results/sub_1061_seeding_mask.nii"
DEFAULT_TRK_FILE_NAME = "result.trk"
DEFAULt_VTK_FILE_NAME = "result.vtk"
DEFAULT_DIR = "C:/Users/HP/Documents/MTP/Slicer-Task/Final Modules/Results/"

class Tractography:
    def __init__(self):
        self.approxMaskPath: str = None
        self.fodfPath: str = None
        self.stepSize: float = 0.2
        self.algo: str = "det"
        self.trkPath : str = None
        self.seedingMaskPath = SEEDING_MASK_FILE_PATH
        self.outputText = None
        self.output_trk_path = None

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
        algos = ["det", "prob", "eudx"]
        self.algo = algos[index]
        slicer.util.showStatusMessage(f"Selected: {self.algo}", 2000)
        print(f"Algorithm set to: {self.algo}")

    # Placeholder for generate, and visualize methods
    def generateTrk(self):
        self._create_binary_mask()
        mask_img = nib.load(self.seedingMaskPath)
        mask_data = get_data_as_mask(mask_img, dtype=bool)

        odf_sh_img = nib.load(self.fodfPath)
        if not np.allclose(np.mean(odf_sh_img.header.get_zooms()[:3]),
                        odf_sh_img.header.get_zooms()[0], atol=1e-03):
            parser.error(
                'ODF SH file is not isotropic. Tracking cannot be ran robustly.')

        if NPV:
            nb_seeds = NPV
            seed_per_vox = True
        elif NT:
            nb_seeds = NT
            seed_per_vox = False
        else:
            nb_seeds = 1
            seed_per_vox = True

        voxel_size = odf_sh_img.header.get_zooms()[0]
        vox_step_size = self.stepSize / voxel_size
        seed_img = nib.load(self.seedingMaskPath)
        seeds = track_utils.random_seeds_from_mask(
            seed_img.get_fdata(dtype=np.float32),
            np.eye(4),
            seeds_count=nb_seeds,
            seed_count_per_voxel=seed_per_vox,
            random_seed=None)

        max_steps = int(MAX_LENGTH / self.stepSize) + 1
        streamlines_generator = LocalTracking(
            self._get_direction_getter(),
            BinaryStoppingCriterion(mask_data),
            seeds, np.eye(4),
            step_size=vox_step_size, max_cross=1,
            maxlen=max_steps,
            fixedstep=True, return_all=True,
            random_seed=SEED,
            save_seeds=SAVE_SEEDS)
        
        scaled_min_length = MIN_LENGTH / voxel_size
        scaled_max_length = MAX_LENGTH / voxel_size

        if SAVE_SEEDS:
            filtered_streamlines, seeds = \
                zip(*((s, p) for s, p in streamlines_generator
                    if scaled_min_length <= length(s) <= scaled_max_length))
            data_per_streamlines = {'seeds': lambda: seeds}
        else:
            filtered_streamlines = \
                (s for s in streamlines_generator
                if scaled_min_length <= length(s) <= scaled_max_length)
            data_per_streamlines = {}

        if COMPRESS:
            filtered_streamlines = (
                compress_streamlines(s, COMPRESS)
                for s in filtered_streamlines)

        tractogram = LazyTractogram(lambda: filtered_streamlines,
                                    data_per_streamlines,
                                    affine_to_rasmm=seed_img.affine)

        
        reference = get_reference_info(seed_img)
        header = create_tractogram_header(FILETYPE, *reference)

        # Use generator to save the streamlines on-the-fly
        if self.trkPath:
            self.output_trk_path = os.path.join(self.trkPath, DEFAULT_TRK_FILE_NAME)
        else:
            self.output_trk_path = os.path.join(DEFAULT_DIR, DEFAULT_TRK_FILE_NAME)
        
        nib.streamlines.save(tractogram, self.output_trk_path, header=header)
        self.outputText.append(f'Trk Genererated Successfully \n (location: {self.output_trk_path}) \n')

    def visualizeTrk(self):
        """Placeholder method for visualizing tractography."""
        if self.output_trk_path == None:
            print("Generate Trk First...")
            return
        tractogram = load_tractogram(self.output_trk_path, reference='same', bbox_valid_check=False, to_space=Space.RASMM)
        print(" tractography Visualization...jdsj")
        if self.trkPath:
            output_vtk_path = os.path.join(self.trkPath, DEFAULt_VTK_FILE_NAME)
        else:
            output_vtk_path = os.path.join(DEFAULT_DIR, DEFAULt_VTK_FILE_NAME)
        
        self._saveStreamlinesVTK(tractogram.streamlines, output_vtk_path )
        reader = vtkPolyDataReader()
        reader.SetFileName(output_vtk_path)
        reader.Update()

        polydata = reader.GetOutput()

        streamline_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        streamline_node.SetAndObservePolyData(polydata)

        display_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode")
        streamline_node.SetAndObserveDisplayNodeID(display_node.GetID())

        display_node.SetColor(0, 1, 0)  
        display_node.SetOpacity(1.0)  

        slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveVolumeID(streamline_node.GetID())
        slicer.app.applicationLogic().PropagateVolumeSelection()
        slicer.app.layoutManager().resetThreeDViews()   
        print("Done Visualization")
        self.outputText.append(f' VTK Files Genererated Succesfully (location : {output_vtk_path}) \n Visualization Complete \n')
    
    # Internal Functions
    def _get_b_matrix(self, order, sphere, sh_basis_type, return_all=False):
        sh_basis = self._honor_authorsnames_sh_basis(sh_basis_type)
        sph_harm_basis = sph_harm_lookup.get(sh_basis)
        if sph_harm_basis is None:
            raise ValueError("Invalid basis name.")
        b_matrix, m, n = sph_harm_basis(order, sphere.theta, sphere.phi)
        if return_all:
            return b_matrix, m, n
        return b_matrix

    def _create_binary_mask(self, threshold=0.5):
        # Load the image
        img = nib.load(self.approxMaskPath)

        # Get the image data
        data = img.get_fdata()

        # Threshold the data to create a binary mask
        mask_data = (data > threshold).astype(np.uint8)

        # Create a new nibabel image with the binary mask
        mask_img = nib.Nifti1Image(mask_data, img.affine, img.header)
        mask_img.set_data_dtype(np.uint8)
        # Save the new image
        nib.save(mask_img, SEEDING_MASK_FILE_PATH)

    def _get_direction_getter(self):
        odf_data = nib.load(self.fodfPath).get_fdata(dtype=np.float32)
        sphere = HemiSphere.from_sphere(get_sphere(SPHERE))
        theta = get_theta(THETA, self.algo)

        non_zeros_count = np.count_nonzero(np.sum(odf_data, axis=-1))
        non_first_val_count = np.count_nonzero(np.argmax(odf_data, axis=-1))

        if self.algo in ['det', 'prob']:
            if non_first_val_count / non_zeros_count > 0.5:
                logging.warning('Input detected as peaks. Input should be'
                                'fodf for det/prob, verify input just in case.')
            if self.algo == 'det':
                dg_class = DeterministicMaximumDirectionGetter
            else:
                dg_class = ProbabilisticDirectionGetter
            return dg_class.from_shcoeff(
                shcoeff=odf_data, max_angle=theta, sphere=sphere,
                basis_type=SH_BASIS,
                relative_peak_threshold=SF_THRESHOLD)
        elif self.algo == 'eudx':
            # Code for type EUDX. We don't use peaks_from_model
            # because we want the peaks from the provided sh.
            odf_shape_3d = odf_data.shape[:-1]
            dg = PeaksAndMetrics()
            dg.sphere = sphere
            dg.ang_thr = theta
            dg.qa_thr = SF_THRESHOLD

            # Heuristic to find out if the input are peaks or fodf
            # fodf are always around 0.15 and peaks around 0.75
            if non_first_val_count / non_zeros_count > 0.5:
                logging.info('Input detected as peaks.')
                nb_peaks = odf_data.shape[-1] // 3
                slices = np.arange(0, 15+1, 3)
                peak_values = np.zeros(odf_shape_3d+(nb_peaks,))
                peak_indices = np.zeros(odf_shape_3d+(nb_peaks,))

                for idx in np.argwhere(np.sum(odf_data, axis=-1)):
                    idx = tuple(idx)
                    for i in range(nb_peaks):
                        peak_values[idx][i] = np.linalg.norm(
                            odf_data[idx][slices[i]:slices[i+1]], axis=-1)
                        peak_indices[idx][i] = sphere.find_closest(
                            odf_data[idx][slices[i]:slices[i+1]])

                dg.peak_dirs = odf_data
            else:
                logging.info('Input detected as fodf.')
                npeaks = 5
                peak_dirs = np.zeros((odf_shape_3d + (npeaks, 3)))
                peak_values = np.zeros((odf_shape_3d + (npeaks, )))
                peak_indices = np.full((odf_shape_3d + (npeaks, )), -1, dtype='int')
                b_matrix = self._get_b_matrix(
                    find_order_from_nb_coeff(odf_data), sphere, SH_BASIS)

                for idx in np.argwhere(np.sum(odf_data, axis=-1)):
                    idx = tuple(idx)
                    directions, values, indices = get_maximas(odf_data[idx],
                                                            sphere, b_matrix,
                                                            SF_THRESHOLD, 0)
                    if values.shape[0] != 0:
                        n = min(npeaks, values.shape[0])
                        peak_dirs[idx][:n] = directions[:n]
                        peak_values[idx][:n] = values[:n]
                        peak_indices[idx][:n] = indices[:n]

                dg.peak_dirs = peak_dirs

            dg.peak_values = peak_values
            dg.peak_indices = peak_indices

            return dg
    
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
          