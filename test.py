import vtk

class MultiVolumeImporterWidget:

    def __init__(self):
        self.__veInitial = 0
        self.__veStep = 1

    def readFODFNIfTI(self, fileName):
        """Reads a 4D fODF NIfTI file and returns the number of frames (Spherical Harmonics components)."""
        print(f'Trying to read {fileName}')
        
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(fileName)
        reader.SetTimeAsVector(True)  
        reader.Update()
        header = reader.GetNIFTIHeader()
        qFormMatrix = reader.GetQFormMatrix()
        
        if not qFormMatrix:
            print(f'Warning: {fileName} does not have a QFormMatrix - using Identity')
            qFormMatrix = vtk.vtkMatrix4x4()

        nFrames = reader.GetTimeDimension()        
        if nFrames == 1:
            print('Warning: Single frame detected, not a multivolume (fODF expected multiple frames).')
            return
        print(f'Successfully read {nFrames} frames (Spherical Harmonics components) from fODF NIfTI file.')

        return nFrames

multiVolumeImporterWidget = MultiVolumeImporterWidget()
nFrames = multiVolumeImporterWidget.readFODFNIfTI('sub-1061__fodf.nii.gz')
print(f'Number of frames in the fODF NIfTI file: {nFrames}')
