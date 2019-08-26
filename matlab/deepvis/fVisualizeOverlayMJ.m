function [ hfig, dImg, dProbOverlay ] = fVisualizeOverlay( sParafile, sModel, iPat, sType, dProbPatch, sPathOutIn, dAlpha, lRot, lMean)
% visualize overlay
if (nargin <9)
    lMean=false;
end
if (nargin <8)
    lRot=false;
end
if(nargin < 7)
    dAlpha = 0.6;%0.4
end

if(ispc)
    sPath = 'W:\ImageSimilarity\Databases\MRPhysics\newProtocol';    
else
    sPath = '/net/linse8-sn/home/s1222/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol';
end
if ~isa(dProbPatch, 'cell')
    dProbPatch={dProbPatch};
end



currpath = fileparts(mfilename('fullpath'));
addpath(genpath([currpath,filesep,'utils',filesep,'export_fig']));
addpath(genpath([currpath,filesep,'utils',filesep,'imoverlay']));

sPats = dir(sPath); 
lMask = cell2mat({sPats(:).isdir}); if(any(~lMask)), sPats(~lMask) = []; end
sPats = sPats(3:end);

[sDataRef,sDataArt,sPathOut] = fGetModelInfo( sModel );
if(exist('sPathOutIn','var'))
    sPathOut = sPathOutIn;
end

% load parameter set
[sPathPara,sFilenamePara] = fileparts(sParafile);
if(~isempty(sPathPara)), cd(sPathPara); end;
eval([sFilenamePara,';']);
if(~isempty(sPathPara)), cd(sCurrPath); end;

%% overlay
if(strcmp(sType,'ref'))
    sDataIn = sDataRef;
elseif(strcmp(sType,'art'))
    sDataIn = sDataArt;
end
fprintf('Loading: %s\n', [sPath,filesep,sPats(iPat).name,filesep,sDataIn]);
dImg = fReadDICOM([sPath,filesep,sPats(iPat).name,filesep,sDataIn]);
iDimImg = size(dImg);

% scaling
dImg = scaleImg(dImg, iScaleRange);
%         dImg = ((dImg - min(dImg(:))) * (range(2)-range(1)))./(max(dImg(:)) - min(dImg(:)));

% patching
fprintf('patching...\n');
if length(patchSize)==2
    [dPatchImg,iPatchSizeImg] = fPatch(dImg, patchSize, patchOverlap);
else%3D-Patching
    [dPatchImg,iPatchSizeImg] = fPatch3D(dImg, patchSize, patchOverlap);
end

dProbOverlay={};
for i=1:length(dProbPatch)
    fprintf('Unpatching %02d/%02d...\n', i, length(dProbPatch));
    if length(patchSize)==2
        dProbOverlay{i} = fUnpatch( dProbPatch{i}, patchSize, patchOverlap, iPatchSizeImg, iDimImg);
    else% 3D UnPatching
        dProbOverlay{i} = fUnpatch3D( dProbPatch{i}, patchSize, patchOverlap, iPatchSizeImg, iDimImg);
    end
end
if lMean
    dProbOverlay=cat(4,dProbOverlay{:});
    dProbVar= var(dProbOverlay,0,4);
    dProbOverlay=mean(dProbOverlay, 4);
end
%TODO rotationen  machen...
%dImg=rot90(dImg,1);
%dProbOverlay{1}=rot90(dProbOverlay{1}, 1);

hfig = [];
%nice col_range[0.32903,1.5],[0,0.57486]
 hfig = fPatchOverlay(dImg,dProbOverlay, [0 1; 0 1], dAlpha, [sPathOut,filesep,'Overlay',filesep,'pat',num2str(iPat)],...
     {[1 size(dImg,2)],[1 size(dImg,1)]},true,false, lRot);

end

