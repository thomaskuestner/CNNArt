function [dImg,dVariance]=fUnpatch3D (dPatch, patchSize, patchOverlap, iZpaddedSize,...
    iActualSize, iClass)
%a new approach of 3D-unpatching
% unpatch to image
% dPatch:   either nPatches x PatchSize(1) x PatchSize(2) x
% iPatchSize(3)-> to create the orig image out of the patches
%           or     nPatches x 2 -> to create a probability map

%
if(nargin < 6), iClass = 1; end
if(nargin < 5), iActualSize = iZpaddedSize; end



dImg= zeros(iZpaddedSize(1),iZpaddedSize(2),iZpaddedSize(3));
iNumVals=zeros(iZpaddedSize(1),iZpaddedSize(2),iZpaddedSize(3));
%stores number of values already taken for the explizit voxel
%the values of different probas are getting stored in a cell for each voxel
%at the end the mean for every voxel is calculated
if length(patchOverlap)==1 
    if (patchOverlap<1)% percentage patching...
        dActSize = round(patchOverlap * patchSize);
    else %same overlap of pixels in each dimension
        dActSize=[patchOverlap, patchOverlap, patchOverlap] ;
    end
elseif length(patchOverlap)==3 %patchOverlap gives the number of pixels going further per patch, in 
    dActSize=patchOverlap;
else 
    exit;%invalid length of Overlap
end

if(size(dPatch,3) == patchSize(1) && size(dPatch,4) == patchSize(2)) % from image patches
    lMode = true;
else % from estimated class labels
    lMode = false;
end

iCorner=[1,1,1];
for iIndex=1:size(dPatch, 1)
    if(lMode)
            tmp = squeeze(dPatch(iIndex,:,:,:,:));
    end
    
    x_range=(iCorner(1):iCorner(1)+patchSize(1)-1);y_range=(iCorner(2):iCorner(2)+patchSize(2)-1);
    z_range=(iCorner(3):iCorner(3)+patchSize(3)-1);
    lMask=false(iZpaddedSize(1),iZpaddedSize(2),iZpaddedSize(3));
    lMask(x_range, y_range, z_range)=true;
    
    if (lMode)%reconstructing the image or getting a probability map...
       dImg(x_range, y_range,z_range)=dImg(x_range, y_range, z_range) + tmp;
    else
        dImg(x_range, y_range,z_range)= dImg(x_range, y_range, z_range)...
            +ones(length(x_range), length(y_range), length(z_range)).* dPatch(iIndex, iClass);
        ;%update the number...
        %of elems in voxels included
    end
    iNumVals(lMask)=iNumVals(lMask) +1;

    
    %update the corner...
    iCorner(3) = iCorner(3) + dActSize(3);%order of filling dImg is important-> depends on fPatch3D
    if(iCorner(3) + patchSize(3)-1 > iZpaddedSize(3))%order =zyx...
        iCorner(3) = 1;
        iCorner(2) = iCorner(2) + dActSize(2);
    end
    if(iCorner(2) + patchSize(2)-1 > iZpaddedSize(2)) % next step in x-Dimension
        iCorner(3) = 1;
        iCorner(2) = 1;
        iCorner(1) = iCorner(1) + dActSize(1);
    end
end
dImg=dImg ./iNumVals;

if(any(iZpaddedSize ~= iActualSize))
    dImg = crop(dImg, iActualSize);
end
dVariance=0;
end

