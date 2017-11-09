function [dPatches, iPadsize] = fPatch(dImg, patchSize, patchOverlap)
% data patching

% (c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de, 2017

if(nargin < 3), patchOverlap = 0.5; end
if(nargin < 2), patchSize = [100 100]; end

%% rigid patching
dPatches = []; 
% zero-padding
dActSize = round((1-patchOverlap) * patchSize);
iPadsize = [ceil(size(dImg,1)/dActSize(1))*dActSize(1), ceil(size(dImg,2)/dActSize(2))*dActSize(2), size(dImg,3)];
dImg = zpad(dImg, iPadsize(1), iPadsize(2), iPadsize(3));  

for iZ=1:size(dImg,3)
    for iX=patchSize(1)/2:dActSize(1):size(dImg,1)-patchSize(1)/2
        for iY=patchSize(2)/2:dActSize(2):size(dImg,2)-patchSize(2)/2
            dPatches = cat(3,dPatches, dImg(iX-patchSize(1)/2+1:iX+patchSize(1)/2, iY-patchSize(2)/2+1:iY+patchSize(2)/2, iZ));
        end
    end
end
        
%% adaptive patching
% TODO: ADD here!