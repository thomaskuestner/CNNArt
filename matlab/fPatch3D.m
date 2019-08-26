function [dPatches, iPadsize] = fPatch3D(dImg, patchSize, patchOverlap)
%patching for 3D CNN
%dImg: 3dimensional image
%input:3D patchsize required e.g. 10x10x10
%returns 4D:dPatches


if length(patchOverlap)==1 
    if (patchOverlap<1)% percentage patching...
        dActSize = round(patchOverlap * patchSize);
    else %same overlap of pixels in each dimension
        dActSize=[patchOverlap, patchOverlap, patchOverlap];
    end
elseif length(patchOverlap)==3 %patchOverlap gives the number of pixels going further per patch
    dActSize=patchOverlap;
else
    %invalid length
    disp('invalid length of patchOverlap')
end
iPadsize = [ceil(size(dImg,1)/dActSize(1))*dActSize(1), ceil(size(dImg,2)/dActSize(2))*dActSize(2),...
    ceil(size(dImg,3)/dActSize(3))*dActSize(3)];

%zero-padding
dImg= zpad(dImg, iPadsize(1), iPadsize(2), iPadsize(3));

dPatches=[];

for iX=patchSize(1)/2:dActSize(1):size(dImg,1)-patchSize(1)/2
    for iY=patchSize(2)/2:dActSize(2):size(dImg,2)-patchSize(2)/2
        for iZ=patchSize(3)/2:dActSize(3):size(dImg,3)-patchSize(3)/2
            %TODO: vektorisierung 
            %range=[iX;iY;iZ] - patchSize ./2 +1:[iX;iY;iZ] -patchSize./2;
            %dPatches = cat(4,dPatches, dImg(range));
            
            iXrange=[iX-patchSize(1)/2+1:iX+patchSize(1)/2];
            iYrange=[iY-patchSize(2)/2+1:iY+patchSize(2)/2];
            iZrange=[iZ-patchSize(3)/2+1:iZ+patchSize(3)/2];
            dPatches = cat(4,dPatches, dImg(iXrange, iYrange, iZrange));
        end
    end
end
