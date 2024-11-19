import os
import json
import nibabel as nib
from dipy.io.streamline import load_tractogram
import numpy as np
from dipy.tracking.metrics import length as slength

class MetricAnalysis:
    def _init_(self):
        self.predictedTrkPath : str = None
        self.groundTruthTrkPath : str = None
        self.diceScore : float = None
        self.overReachScore: float = None
        self.overlapScore: float = None
        self.ui = None

    @staticmethod
    def isValidPath(path: str) -> bool:
        """Validate if the provided path exists and is a file."""
        return os.path.isfile(path)

    # Setter for predictedTrkPath
    def setPredictedTrkPath(self, path: str):
        """Set the path for the predicted .trk file after validation."""
        if self.isValidPath(path):
            self.predictedTrkPath = path
            print(f"Predicted .trk path set to: {self.predictedTrkPath}")
        else:
            raise FileNotFoundError(f"Invalid predicted .trk path: {path}")

    # Setter for groundTrkPath
    def setGroundTruthTrkPath(self, path: str):
        """Set the path for the ground truth .trk file after validation."""
        if self.isValidPath(path):
            self.groundTruthTrkPath = path
            print(f"Ground truth .trk path set to: {self.groundTruthTrkPath}")
        else:
            raise FileNotFoundError(f"Invalid ground truth .trk path: {path}")

    def set_dice_score(self, score: str):
        self.diceScore = score

    def set_overlap_score(self, score: str):
        self.overlapScore = score

    def set_overreach_score(self, score: str):
        self.overreachScore = score
    
    def generateMetrics(self):
        print("Generating Metrics ...")
        if self.predictedTrkPath is None or self.groundTruthTrkPath is None:
            print("Both predicted_trk and groundTruthTrkPath are required.")
            return

        def compute_binary_map(sft):
            """Compute the binary map of a tractogram."""
            sft.to_vox()
            sft.to_corner()
            streamlines = sft.streamlines
            _, dimensions, _, _ = sft.space_attributes
            if not streamlines:
                pass
            binary_map = self.compute_tract_counts_map(streamlines, dimensions)
            binary_map[binary_map > 0] = 1
            return binary_map

        def compute_voxel_pairwise_measures(bundle_fname, gs_fname):
            """Compute comparison measures between two bundles."""
            bundle_binary_map = compute_binary_map(bundle_fname)
            gs_binary_map = compute_binary_map(gs_fname)

            bundle_indices = np.where(bundle_binary_map.flatten() > 0)[0]
            gs_indices = np.where(gs_binary_map.flatten() > 0)[0]

            tp = len(np.intersect1d(bundle_indices, gs_indices))
            fp = len(np.setdiff1d(bundle_indices, gs_indices))
            fn = len(np.setdiff1d(gs_indices, bundle_indices))

            if tp == 0:
                overlap = 0.
                overreach = None
                dice = 0.
            else:
                overlap = tp / float(tp + fn)
                overreach = fp / float(tp + fn)
                dice = 2 * tp / float(2 * tp + fp + fn)

            return {"dice": dice, "overlap": overlap, "overreach": overreach}
        
        scores = compute_voxel_pairwise_measures(load_tractogram(self.predictedTrkPath, 'same', bbox_valid_check=False), load_tractogram(self.groundTruthTrkPath, 'same', bbox_valid_check=False))
        self.diceScore = scores["dice"]
        self.overlap = scores["overlap"]
        self.overreach = scores['overreach']
        self.ui.diceScore.setText(f'Dice Score: {scores["dice"]}')
        self.ui.overlapScore.setText(f'Overlap: {scores["overlap"]}')
        self.ui.overreachScore.setText(f'Overreach: {scores["overreach"]}')

    def compute_tract_counts_map(self, streamlines, vol_dims):
        """Compute the number of different tracks going through each voxel."""
        # Set numpy error handling
        flags = np.seterr(divide="ignore", under="ignore")

        # Convert volume dimensions to integers
        vol_dims = np.asarray(vol_dims).astype(int)
        n_voxels = np.prod(vol_dims)

        # Initialize traversal tags
        traversal_tags = np.zeros((n_voxels,), dtype=int)
        touched_tags = np.zeros((n_voxels,), dtype=int)

        streamlines_len = len(streamlines)

        if streamlines_len == 0:
            np.seterr(**flags)
            return traversal_tags.reshape(vol_dims)

        # Initialize arrays for points and direction vectors
        in_pt = np.zeros(3, dtype=np.double)
        next_pt = np.zeros(3, dtype=np.double)
        dir_vect = np.zeros(3, dtype=np.double)
        cur_edge = np.zeros(3, dtype=np.double)
        cur_voxel_coords = np.zeros(3, dtype=int)

        # Dimensions for voxel space
        vd = list(vol_dims)
        x_slice_size = vd[1] * vd[2]

        for track_idx in range(streamlines_len):
            t = streamlines[track_idx].astype(np.double)

            for pno in range(t.shape[0] - 1):
                for cno in range(3):
                    in_pt[cno] = t[pno, cno]
                    next_pt[cno] = t[pno + 1, cno]
                    dir_vect[cno] = next_pt[cno] - in_pt[cno]
                    cur_edge[cno] = in_pt[cno]

                dir_vect_norm = norm(dir_vect[0], dir_vect[1], dir_vect[2])

                if dir_vect_norm == 0:
                    continue

                remaining_dist = dir_vect_norm

                # Check if it's already a real edge, if not find the closest edge
                if (floor(cur_edge[0]) != cur_edge[0] and
                    floor(cur_edge[1]) != cur_edge[1] and
                    floor(cur_edge[2]) != cur_edge[2]):
                    cur_edge = get_closest_edge(in_pt[0], in_pt[1], in_pt[2],
                                                dir_vect[0], dir_vect[1], dir_vect[2])

                while True:
                    length_ratio = inf
                    for cno in range(3):
                        if dir_vect[cno] != 0:
                            length_ratio = min(fabs((cur_edge[cno] - in_pt[cno]) / dir_vect[cno]), length_ratio)

                    remaining_dist -= length_ratio * dir_vect_norm

                    if remaining_dist < 0 and not fabs(remaining_dist) < 1e-8:
                        break

                    for cno in range(3):
                        cur_voxel_coords[cno] = int(floor(in_pt[cno] + 0.5 * length_ratio * dir_vect[cno]))

                    el_no = (cur_voxel_coords[0] * x_slice_size +
                            cur_voxel_coords[1] * vd[2] +
                            cur_voxel_coords[2])

                    if touched_tags[el_no] != track_idx + 1:
                        touched_tags[el_no] = track_idx + 1
                        traversal_tags[el_no] += 1

                    for cno in range(3):
                        in_pt[cno] += length_ratio * dir_vect[cno]
                        if fabs(in_pt[cno]) <= 1e-16:
                            in_pt[cno] = 0.0

                    cur_edge = get_closest_edge(in_pt[0], in_pt[1], in_pt[2],
                                                dir_vect[0], dir_vect[1], dir_vect[2])

            # Add last point
            for cno in range(3):
                cur_voxel_coords[cno] = int(floor(in_pt[cno] + 0.5 * (next_pt[cno] - in_pt[cno])))

            el_no = (cur_voxel_coords[0] * x_slice_size +
                    cur_voxel_coords[1] * vd[2] +
                    cur_voxel_coords[2])

            if touched_tags[el_no] != track_idx + 1:
                touched_tags[el_no] = track_idx + 1
                traversal_tags[el_no] += 1

        np.seterr(**flags)
        return traversal_tags.reshape(vol_dims)


from math import sqrt, floor, ceil, fabs, inf
import numpy as np

def norm(x, y, z):
        """Compute the Euclidean norm of a 3D vector."""
        return sqrt(x * x + y * y + z * z)

def get_closest_edge(p_x, p_y, p_z, d_x, d_y, d_z, eps=1.):
        """Get the closest edge to the point based on the direction vector."""
        edge = np.zeros(3)
        edge[0] = floor(p_x + eps) if d_x >= 0.0 else ceil(p_x - eps)
        edge[1] = floor(p_y + eps) if d_y >= 0.0 else ceil(p_y - eps)
        edge[2] = floor(p_z + eps) if d_z >= 0.0 else ceil(p_z - eps)
        return edge