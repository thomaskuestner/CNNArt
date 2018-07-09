
from utilsGUI.Label import Label

class Dataset:
    def __init__(self, pathdata, pathlabel=None, artefact=None, bodyregion=None, tWeighting=None):
        self.pathdata = pathdata
        self.artefact = artefact
        self.bodyregion = bodyregion
        self.mrtWeighting = tWeighting

        if pathlabel==None:
            if tWeighting == None:
                if artefact == 'ref' and bodyregion =='head':
                    self.datasetLabel = Label.getLabel(Label.HEAD, Label.REFERENCE)
                elif artefact == 'motion' and bodyregion == 'head':
                    self.datasetLabel = Label.getLabel(Label.HEAD, Label.MOTION)
                elif artefact == 'ref' and bodyregion == 'abdomen':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.REFERENCE)
                elif artefact == 'motion' and bodyregion == 'abdomen':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.MOTION)
                elif artefact == 'shim' and bodyregion == 'abdomen':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.SHIM)
                elif artefact == 'ref' and bodyregion == 'pelvis':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.REFERENCE)
                elif artefact == 'motion' and bodyregion == 'pelvis':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.MOTION)
                elif artefact == 'shim' and bodyregion == 'pelvis':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.SHIM)
                else:
                    raise ValueError('Problem with dataset labeling!')
            else:
                if artefact == 'ref' and bodyregion =='head' and tWeighting == 't1':
                    self.datasetLabel = Label.getLabel(Label.HEAD, Label.REFERENCE, Label.T1)
                elif artefact == 'motion' and bodyregion == 'head' and tWeighting == 't1':
                    self.datasetLabel = Label.getLabel(Label.HEAD, Label.MOTION, Label.T1)
                elif artefact == 'ref' and bodyregion == 'abdomen' and tWeighting == 't1':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.REFERENCE, Label.T1)
                elif artefact == 'motion' and bodyregion == 'abdomen' and tWeighting == 't1':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.MOTION, Label.T1)
                elif artefact == 'ref' and bodyregion == 'abdomen' and tWeighting == 't2':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.REFERENCE, Label.T2)
                elif artefact == 'shim' and bodyregion == 'abdomen' and tWeighting == 't2':
                    self.datasetLabel = Label.getLabel(Label.ABDOMEN, Label.SHIM, Label.T2)
                elif artefact == 'ref' and bodyregion == 'pelvis' and tWeighting == 't1':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.REFERENCE, Label.T1)
                elif artefact == 'ref' and bodyregion == 'pelvis' and tWeighting == 't2':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.REFERENCE, Label.T2)
                elif artefact == 'motion' and bodyregion == 'pelvis' and tWeighting == 't1':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.MOTION, Label.T1)
                elif artefact == 'motion' and bodyregion == 'pelvis' and tWeighting == 't2':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.MOTION, Label.T2)
                elif artefact == 'shim' and bodyregion == 'pelvis' and tWeighting == 't2':
                    self.datasetLabel = Label.getLabel(Label.PELVIS, Label.SHIM, Label.T2)
                else:
                    raise ValueError('Problem with dataset labeling!')
        else:
            self.datasetLabel = pathlabel;

    def getPathdata(self):
        return self.pathdata

    def getDatasetLabel(self):
        return self.datasetLabel

    def getArtefact(self):
        return self.artefact

    def getBodyRegion(self):
        labelBodyRegion = None
        if self.bodyregion == 'head':
            labelBodyRegion = Label.HEAD
        elif self.bodyregion == 'abdomen':
            labelBodyRegion = Label.ABDOMEN
        elif self.bodyregion == 'pelvis':
            labelBodyRegion = Label.PELVIS

        return self.bodyregion, labelBodyRegion

    def getMRTWeighting(self):
        tWeighting = None
        if self.mrtWeighting == 't1':
            tWeighting = Label.T1
        elif self.mrtWeighting == 't2':
            tWeighting = Label.T2
        return self.mrtWeighting, tWeighting
