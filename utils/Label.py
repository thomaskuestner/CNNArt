"""
@author: Yannick Wilhelm
@email: yannick.wilhelm@gmx.de
@date: January 2018
"""

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