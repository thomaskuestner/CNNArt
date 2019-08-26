function [dDeepVis, dSubsets] = fVisualizePoint( sPatsIn, sModel, sParafile, lType )
% visualize model

% (c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de, 2017

if(nargin < 4)
    lType = true; % true = artefact, false = reference
end

% database
if(ispc)
    sPath = 'W:\ImageSimilarity\Databases\MRPhysics\newProtocol';    
    sPath = 'D:\IS\MRPhysics\newProtocol';
else
    sPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol';
end

% get current path
sCurrPath = fileparts(mfilename('fullpath'));
if(isempty(sCurrPath))
    sCurrPath = '/net/linse8-sn/home/m123/Matlab/ArtefactClassification';
end
addpath(genpath(sCurrPath));

[sDataRef,sDataArt,sOutpath,sOptiModels] = fGetModelInfo( sModel );

% prepare inputs
if(ischar(sPatsIn))
    sPathDICOMs = {sPatsIn};
    [sPats, sFile] = fileparts(sPatsIn);
    [~,sPats] = fileparts(sPats);  
    sPats = {sPats};
    sSaveName = [sFile,'_visOut'];
else
    sPats = dir(sPath); 
    lMask = cell2mat({sPats(:).isdir}); if(any(~lMask)), sPats(~lMask) = []; end
    sPats = sPats(3:end);
    sPats = sPats(sPatsIn);
    sPats = {sPats(:).name};
    if(lType)
        sPathDICOMs = cellfun(@(x) [sPath, filesep, x, filesep, sDataArt], sPats, 'UniformOutput', false);
        sSaveName = ['art_visOut'];
    else
        sPathDICOMs = cellfun(@(x) [sPath, filesep, x, filesep, sDataRef], sPats, 'UniformOutput', false);
        sSaveName = ['ref_visOut'];
    end
end

% load parameter set
[sPathPara,sFilenamePara] = fileparts(sParafile);
if(~isempty(sPathPara)), cd(sPathPara); end;
eval([sFilenamePara,';']);
if(~isempty(sPathPara)), cd(sCurrPath); end;

if(patchSize == [30 30])
    sOptiModel = sOptiModels{1};
elseif(patchSize == [40 40])   
    sOptiModel = sOptiModels{2};
elseif(patchSize == [60 60]);
    sOptiModel = sOptiModels{3};
end

%% visualize it
for iPat=1:length(sPats)
    imgPred = fReadDICOM(sPathDICOMs{iPat});
    [dPatches,iPadSize] = fPatch(imgPred, patchSize, patchOverlap);
    X_test =  permute(dPatches,[3 4 1 2]);
    y_test = zeros(size(X_test,1),1); % 0 = reference, 1 = artifact => but fake it to interesting class -> hardcode it to "0"
    
    sSavefile = [sOutpath,filesep,sPats{iPat},'_',sSaveName];
    save([sOutpath,filesep,'visualize.mat'], 'X_test', 'y_test', 'sSavefile', 'sOptiModel', 'patchSize');

    system(sprintf(['python2 fVisualizePoint.py -i %s'], sOutpath));
    copyfile([sOutpath,filesep,'visualize.mat'], [sSavefile(1:end-3),'In']);
end

%% load results
dSubsets = cell(1,length(sPats));
dDeepVis = cell(1,length(sPats));
for iPat=1:length(sPats)
    sSavefile = [sOutpath,filesep,sPats{iPat},'_',sSaveName];
    load([sSavefile,'_DV.mat']);
    dDeepVis{iPat} = resultDV;
    
    if(exist([sSavefile,'_SS.mat'],'file'))
        load([sSavefile,'_SS.mat']);
        sCmd = ['results = cat(1,'];
        for iI=1:size(resultSS,2)
            sCmd = [sCmd,sprintf('resultSS(:,%d,1,:,:),',iI)];
        end
        sCmd = [sCmd(1:end-1),');'];
        eval(sCmd);
        results = squeeze(results);
        dSubsets{iPat} = fUnpatch(results,patchSize,patchOverlap, iPadSize, size(imgPred));
    end
end

