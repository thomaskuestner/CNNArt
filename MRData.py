class MRData:
    sPath = ''
    sBodyregion = ''
    sArtefact = ''
    iNumber = []

    def __init__(self,sPath,sBodyregion,sArtefact,iNumber):
        self.sPath = sPath
        self.sBodyregion = sBodyregion
        self.sArtefact = sArtefact
        self.iNumber = iNumber

    def isArt(self):