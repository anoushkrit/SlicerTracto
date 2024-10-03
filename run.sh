# pip3 install nibabel numpy

python3 /Users/mahir/Desktop/MTP/toint.py \
    /Users/mahir/Desktop/MTP/data/sub-1061-generated_approximated_mask_1mm.nii.gz \
    /Users/mahir/Desktop/MTP/data/AF_L_mrm_int.nii.gz

# echo "1061"
# python /home/turing/TrackToLearn/scilpy/scripts/scil_compute_local_tracking.py \
#     /datasets/tracto_ashutosh/Github_classicals/data/sub-1061/FODF_Metrics/sub-1061__fodf.nii.gz \
#     /datasets/tracto_ashutosh/Github_classicals/data/sub-1061/Seeding_Mask/AF_L_mrm_int.nii.gz \
#     /datasets/tracto_ashutosh/Github_classicals/data/sub-1061/Seeding_Mask/AF_L_mrm_int.nii.gz  \
#     /datasets/tracto_ashutosh/Github_classicals/out_AFL/trk_1061_mrm_detT.trk \
#     --step 0.2 --algo det --npv 7 -v