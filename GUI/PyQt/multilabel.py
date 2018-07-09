"""
@author: Yannick Wilhelm
@email: yannick.wilhelm@gmx.de
@date: January 2018
"""
import numpy as np

class Label:
    '''
    class Label has a static method getLabel(bodyRegion, artefact) to return the specific label for
    an artefact and a body region. The artefacts and body region labels can be accessed with static members of class
    label.
    '''

    # Artefact labels
    REFERENCE = 0
    MOTION = 1
    SHIM = 2
    NOISE = 3
    MOTION_AND_SHIM = 4
    MOTION_AND_NOISE = 5
    SHIM_AND_NOISE = 6
    MOTION_AND_SHIM_AND_NOISE = 7

    # Body Region labels
    HEAD = 10
    ABDOMEN = 20
    PELVIS = 30

    # MRI weighting label
    T1 = 100
    T2 = 200

    @staticmethod
    def getLabel(bodyRegion, artefact, tWeighting = 0):
        if bodyRegion==Label.HEAD or bodyRegion == Label.ABDOMEN or bodyRegion == Label.PELVIS:
            if artefact<=7 and artefact>=0:
                if tWeighting == 0:
                    #no MRI weighting
                    return bodyRegion + artefact
                elif tWeighting == Label.T1 or Label.T2:
                    return bodyRegion + artefact + tWeighting
                else:
                    raise ValueError('No valid MRI weighting')
            else:
                raise ValueError('No valid Artefact!')
        else:
            raise ValueError('No valid Body Region!')


    @staticmethod
    def mapClassesToOutputVector(classes, usingArtefacts=True, usingBodyRegion=True, usingTWeightings=True):
        '''
        computes the output vector of used classes considering the positional number of the labels given by used sublabels. For example:
        classes: [110, 111, 120, 121, 220, 221]
        for positionalNumber=3 (usingArtefacts=True, usingBodyRegion=True, usingTWeightings=True):
            {'110': [1 0 0 0 0 0],
             '111': [0 1 0 0 0 0],
             ....
             '221': [0 0 0 0 0 1]}
        for positionalNumber = 2: the MRT t-weightning will be not considered (usingArtefacts=True, usingBodyRegion=True, usingTWeightings=false)
            {'110': [1 0 0 0],
             '111': [0 1 0 0],
             '120': [0 0 1 0],
             '121': [0 0 0 1],
             '220': [0 0 1 0],
             '221': [0 0 0 1]}
        :param classes: vector with all used class labels
        :param positionalNumber: positional number of labels, on which the classes are mapped to output vector
        :return: dict with mapped outputvectors
        '''

        if usingArtefacts == True and usingBodyRegion == True and usingTWeightings == True:
            classMappings = {}
            i = 0
            for uniqueClass in classes:
                outputVec = np.zeros([len(classes)])
                outputVec[i] = 1
                classMappings[uniqueClass] = outputVec
                i += 1

        if usingArtefacts == True and usingBodyRegion == True and usingTWeightings == False:
            # MRI t weighting is not considered
            classesReduced = classes%100
            uniqueClassesReduced = np.asarray(np.unique(classesReduced), dtype=int)

            subMappings = {}
            j = 0
            for uniqueClassReduced in uniqueClassesReduced:
                reducedOutputVec = np.zeros([len(uniqueClassesReduced)])
                reducedOutputVec[j] = 1
                subMappings[uniqueClassReduced] = reducedOutputVec
                j+=1

            classMappings = {}
            i = 0
            for uniqueClass in classes:
                classMappings[classes[i]] = subMappings[classesReduced[i]]
                i += 1

        if usingArtefacts == True and usingBodyRegion == False and usingTWeightings == False:
            # MRI t weighting and body regions are not considered
            classesReduced = classes%100
            classesReduced = classesReduced%10
            uniqueClassesReduced = np.asarray(np.unique(classesReduced), dtype=int)

            subMappings = {}
            j=0
            for uniqueClassReduced in uniqueClassesReduced:
                reducedOutputVec = np.zeros([len(uniqueClassesReduced)])
                reducedOutputVec[j] = 1
                subMappings[uniqueClassReduced] = reducedOutputVec
                j+=1

            classMappings = {}
            i = 0
            for uniqueClass in classes:
                classMappings[classes[i]] = subMappings[classesReduced[i]]
                i += 1

        return classMappings