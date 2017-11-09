function [ hfig, dImg, dProbOverlay ] = fVisualizeOverlay( sParafile, sModel, iPat, sType, dProbPatch, sPathOutIn, dAlpha )
% visualize overlay

if(nargin < 7)
    dAlpha = 0.4;
end

if(ispc)
    sPath = 'W:\ImageSimilarity\Databases\MRPhysics\newProtocol';  
    sPath = 'D:\IS\MRPhysics\newProtocol';
else
    sPath = '/scratch/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol';
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
fprintf('Unpatching...\n');
[dPatchImg,iPatchSizeImg] = fPatch(dImg, patchSize, patchOverlap);

dProbOverlay = fUnpatch( dProbPatch, patchSize, patchOverlap, iPatchSizeImg, iDimImg);

hfig = [];
% hfig = fPatchOverlay(dImg,dProbOverlay, [0 0.5; 0 1], dAlpha, [sPathOut,filesep,'Overlay',filesep,num2str(iPat)]);

end

