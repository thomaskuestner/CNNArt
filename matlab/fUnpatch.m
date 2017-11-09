function dImg = fUnpatch( dPatch, patchSize, patchOverlap, iZpaddedSize, iActualSize, iClass )
% unpatch to image
% dPatch:   either nPatches x iPatchSize(1) x iPatchSize(2)
%           or     nPatches x 2

% (c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de, 2017

if(nargin < 6), iClass = 1; end
if(nargin < 5), iActualSize = iZpaddedSize; end

% check input
if(ndims(dPatch) > 3)
    error('Invalid Patch size input');
end
if(size(dPatch,1) == patchSize(1) && size(dPatch,2) == patchSize(2)) % patches in first two dimensions
    dPatch = permute(dPatch,[3 1 2]);    
elseif(size(dPatch,1) == patchSize(2) && size(dPatch,2) == patchSize(1)) % patches in first two dimensions
    dPatch = permute(dPatch,[3 2 1]);
end

% init for resulting image
rows_recon = iZpaddedSize(1); cols_recon = iZpaddedSize(2); layers_recon = iZpaddedSize(3); 
dImg=zeros(rows_recon,cols_recon,layers_recon);
dImgLayer=zeros(rows_recon,cols_recon);

% dPatch_rou = round(dPatch);
if(size(dPatch,2) == patchSize(1) && size(dPatch,3) == patchSize(2)) % from image patches
    lMode = true;
else % from estimated class labels
    lMode = false;
end
    
count_row=1;
count_col=1;
count_layer=1;
lFilled=false(rows_recon,cols_recon); % overlay just possible inside slice
for iIdx=1:size(dPatch,1)
    if(lMode)
        tmp = squeeze(dPatch(iIdx,:,:));
    else
        tmp = ones(patchSize(1),patchSize(2))*dPatch(iIdx,iClass);
    end
    lMask = false(rows_recon,cols_recon); lMask(max([1,count_row]):min([count_row+patchSize(1)-1,rows_recon]),max([1,count_col]):min([count_col+patchSize(2)-1,cols_recon])) = true;
    lMean = lMask & lFilled;
    iX = max([1,count_row]):min([count_row+patchSize(1)-1,rows_recon]);
    iY = max([1,count_col]):min([count_col+patchSize(2)-1,cols_recon]);
    dImgLayer(iX,iY) = dImgLayer(iX,iY) + tmp(1:length(iX),1:length(iY));
    dImgLayer(lMean) = 0.5.*dImgLayer(lMean);
    lFilled = lFilled | lMask;

    count_col = count_col + patchOverlap .* patchSize(2);
    if(count_col + patchSize(2)-1 > cols_recon)
        count_col = 1;
        count_row = count_row + patchOverlap .* patchSize(1);
    end
    if(count_row + patchSize(1)-1 > rows_recon) % next layer
        dImg(:,:,count_layer) = dImgLayer;
        count_col = 1;
        count_row = 1;
        count_layer = count_layer + 1;
        lFilled=false(rows_recon,cols_recon);
        dImgLayer=zeros(rows_recon,cols_recon);
    end
end


if(any(iZpaddedSize ~= iActualSize))
    dImg = crop(dImg, iActualSize);
end

end

