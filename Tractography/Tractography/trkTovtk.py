import vtk
from dipy.io.streamline import load_tractogram
from dipy.io.stateful_tractogram import Space
from dipy.io.utils import is_header_compatible

def saveStreamlinesVTK(streamlines, pStreamlines):
    
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

# Load the tractogram file with a reference to its own header for space.
tractogram = load_tractogram("/Users/mahir/Desktop/MTP/trk_1061_mrm_detT.trk", reference='same', bbox_valid_check=False, to_space=Space.RASMM)

# Save the streamlines as a VTK file
saveStreamlinesVTK(tractogram.streamlines, "sl.vtk")
