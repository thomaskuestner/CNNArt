class MRData:
    # full path is given by DatabaseInfo.sPathIn + patient + sPath
    sPath = '' # name of DICOM directory
    sPathlabel = ''
    sBodyregion = ''
    sArtefact = ''
    sContrast = ''
    sSeq = ''
    sNumber = []

    def __init__(self,sPath,sPathlabel,sArtefact,sBodyregion):
        self.sPath = sPath
        self.sPathlabel = sPathlabel
        self.sBodyregion = sBodyregion
        self.sArtefact = sArtefact

        strtmp = sPath.split('_')
        self.sContrast = strtmp[0]
        self.sSeq = strtmp[1]
        self.sNumber = strtmp[-1]

    @property
    def isart(self):
        if self.sArtefact == 'ref':
            return false
        else:
            return true

