function fTrainCNN(sParafile, sModel, lPats, iGPU)
% inputs
% sParafile     parameter file
% sModel        CNN model
% lPats         logical array for patients used to run cross-validation -> parallization
% iGPU          used GPU

% (c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de, 2017

%% train network
% database
if(ispc)
    sPath = 'W:\ImageSimilarity\Databases\MRPhysics\newProtocol';    
else
    sPath = '/scratch/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol';
end

sCurrPath = fileparts(mfilename('fullpath'));
if(isempty(sCurrPath))
    sCurrPath = '/net/linse8-sn/home/m123/Matlab/ArtefactClassification';
end
addpath(genpath(sCurrPath));

% parse available patients
% sPats = {'ab', 'dc', 'fg', 'hr', 'hs', 'jw', 'ma', 'ms', 'sg', 'yb'};
sPats = dir(sPath); 
lMask = cell2mat({sPats(:).isdir}); if(any(~lMask)), sPats(~lMask) = []; end
sPats = sPats(3:end);

if(nargin < 4 || ~exist('iGPU','var'))
    iGPU = 2;
end

if(nargin < 3 || ~exist('lPats','var'))
    lPats = true(length(sPats));
end

if(nargin < 2 || ~exist('sModel', 'var'))
    % cnn model for: 'motion_head' | 'motion_abd' | 'shim' | 'noise'
    sModel = 'motion_head';
end

if(nargin < 1 || ~exist('sParafile', 'var'))
    sParafile = 'parameters_default.m';
end

[sDataRef,sDataArt,sPathOut] = fGetModelInfo( sModel );

if(ischar(sDataRef))
	sDataRef = {sDataRef;0};
end
if(ischar(sDataArt))
	sDataArt = {sDataArt;1};
end
sDataAll = cat(2,sDataRef,sDataArt);

% load parameter set
[sPathPara,sFilenamePara] = fileparts(sParafile);
if(~isempty(sPathPara)), cd(sPathPara); end;
eval([sFilenamePara,';']);
if(~isempty(sPathPara)), cd(sCurrPath); end;

%% data patching
allPatches = [];
allY = [];
iPats = [];

for iPat = 1:length(sPats)
    fprintf('Pat %d/%d', iPat, length(sPats));
    if(strcmp(sPats(iPat).name,'cb')) % !!!!!!!!!!!!!! >> Data missing
        continue;
    end
    for iJ=1:size(sDataAll,2) % ref and artefact
        fprintf('.');
        %if(iJ == 1)
        %    sDataIn = sDataRef;
        %else
        %    sDataIn = sDataArt;
        %end
		sDataIn = sDataAll{1,iJ};
        dImg = fReadDICOM([sPath,filesep,sPats(iPat).name,filesep,sDataIn]);
        [nX,nY,nZ] = size(dImg);

        % scaling
        dImg = scaleImg(dImg, iScaleRange);
%         dImg = ((dImg - min(dImg(:))) * (range(2)-range(1)))./(max(dImg(:)) - min(dImg(:)));
       
        dimension = size(dImg);
        
        % patching
        dPatches = fPatch(dImg, patchSize, patchOverlap);
        
		y = sDataAll{2,iJ} .* ones(size(dPatches,3),1);
        %if(iJ == 1) % ref
        %    y = zeros(size(dPatches,3),1);
        %else
        %    y = ones(size(dPatches,3),1);
        %end
        
        allPatches = cat(3,allPatches,dPatches);
        allY = cat(1,allY,y);
        iPats = cat(1,iPats,iPat * ones(size(dPatches,3),1));
    end
    fprintf('\n');
end

dimensions = [dimension(3), 1, dimension(1),dimension(2)];

%% split training and test set
if(strcmp(sSplitting,'normal'))
    nPatches = size(allPatches,3);
    dVal = floor(dSplitval* nPatches);

    rand_num = randperm(nPatches,dVal);
    X_test = allPatches(:,:,rand_num);
    y_test = allY(rand_num);

    X_train = allPatches;
    X_train(:,:,rand_num)=[]; 
    y_train = allY;
    y_train(rand_num)=[];

    % arrange data for usage in python and keras
    X_train = permute(X_train,[3 4 1 2]);
    X_test = permute(X_test,[3 4 1 2]);
    % X_per = permute(allPatches,[3 4 1 2]);
    
    % save for python
    sPathMat = [sPathOut,filesep,'normal_', num2str(patchSize(1)), num2str(patchSize(2)),'.mat'];
    save(sPathMat, 'X_train', 'X_test', 'y_train', 'y_test', 'dimensions', 'patchSize', 'patchOverlap', '-v7.3');

    % call python
    fSetGPU( iGPU );
    system(sprintf('python2 cnn_main.py -i %s -o %s -m %s -t -p %s', sPathMat, [sPathOut,filesep,'out_normal'], sModel, sOpti));
elseif(strcmp(sSplitting,'crossvalidation_data'))
    if(exist([sPathOut,filesep, num2str(patchSize(1)), num2str(patchSize(2)), filesep, 'data',filesep, 'iFolds.mat'], 'file'))
        load([sPathOut,filesep, num2str(patchSize(1)), num2str(patchSize(2)), filesep, 'data',filesep, 'iFolds.mat']);
    else
        iInd = crossvalind('Kfold', size(allPatches,3), nFolds);
        save([sPathOut,filesep, num2str(patchSize(1)), num2str(patchSize(2)), filesep, 'data',filesep, 'iFolds.mat'], 'iInd');
    end
    for iFold = 1:nFolds
        if(~lPats(iFold)), continue; end;
        fprintf('Fold %d\n', iFold);
        lMask = iInd == iFold;
        
        X_test = allPatches(:,:,lMask);
        y_test = allY(lMask);
        
        X_train = allPatches(:,:,~lMask);
        y_train = allY(~lMask);
        
        X_train = permute(X_train,[3 4 1 2]);
        X_test = permute(X_test,[3 4 1 2]);
        
        % save for python
        sPathOutCurr = [sPathOut,filesep, num2str(patchSize(1)), num2str(patchSize(2)), filesep, 'data',filesep, num2str(iFold,'%02d')];
        if(~exist(sPathOutCurr,'dir'))
            mkdir(sPathOutCurr);
        end
        sFiles = dir(sPathOutCurr); if(length(sFiles(3:end)) == prod(structfun(@(x) length(x), sOptiPara))*3 + 1), continue; end % learning rates, batch sizes, 3 output files, 1 input file

        sPathMat = [sPathOutCurr,filesep,'crossVal_data',num2str(iFold,'%02d'),'_', num2str(patchSize(1)), num2str(patchSize(2)),'.mat'];
        save(sPathMat, 'X_train', 'X_test', 'y_train', 'y_test', 'dimensions', 'patchSize', 'patchOverlap', '-v7.3');
        
        % call python
        fSetGPU( iGPU );
        system(sprintf('python2 cnn_main.py -i %s -o %s -m %s -t -p %s', sPathMat, [sPathOutCurr,filesep,'outcrossVal_data',num2str(iFold,'%02d')], sModel, sOpti));

    end
    
    
elseif(strcmp(sSplitting,'crossvalidation_patient'))
    for iPat = 1:length(sPats)
        if(~lPats(iPat)), continue; end;
        fprintf('Patient %d\n', iPat);
        lMask = iPats == iPat;
        
        X_test = allPatches(:,:,lMask);
        y_test = allY(lMask);
        
        X_train = allPatches(:,:,~lMask);
        y_train = allY(~lMask);
        
        X_train = permute(X_train,[3 4 1 2]);
        X_test = permute(X_test,[3 4 1 2]);
        
        % save for python
        sPathOutCurr = [sPathOut,filesep, num2str(patchSize(1)), num2str(patchSize(2)), filesep, 'patient', filesep, num2str(iPat,'%02d')];
        if(~exist(sPathOutCurr,'dir'))
            mkdir(sPathOutCurr);
        end
        sFiles = dir(sPathOutCurr); if(length(sFiles(3:end)) == prod(structfun(@(x) length(x), sOptiPara))*3 + 1), continue; end % learning rates, batch sizes, 3 output files, 1 input file

        sPathMat = [sPathOutCurr,filesep,'crossVal',num2str(iPat,'%02d'),'_', num2str(patchSize(1)), num2str(patchSize(2)),'.mat'];
        save(sPathMat, 'X_train', 'X_test', 'y_train', 'y_test', 'dimensions', 'patchSize', 'patchOverlap', '-v7.3');
        
        % call python
        fSetGPU( iGPU );
        system(sprintf('python2 cnn_main.py -i %s -o %s -m %s -t -p %s', sPathMat, [sPathOutCurr,filesep,'outcrossVal',num2str(iPat,'%02d')], sModel, sOpti));
    end
end


end
