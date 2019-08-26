function  fCreateAndSavePatches( sParafile, sModel, iPatchSize, sOutPath, lPats, iRots)
%FCREATEANDSAVEPATCHES3D Summary of this function goes here
%   Detailed explanation goes here
%only for normal, crossvalpat and none right now


if ~isa(sModel, 'cell')
    sModel={sModel};
end
if ~isa(iRots, 'cell')
    iRots={iRots} ;
end
%exit if data already exist
if exist(sOutPath, 'file')
    %exit;
    sprintf('Data already exist!!')
    return
end
sPath='/net/linse8-sn/home/s1222/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol';
if (length(iPatchSize)==2);
    l3D=false;%determine if 3D
else l3D=true; end;

sCurrPath = fileparts(mfilename('fullpath'));
if(isempty(sCurrPath))
    sCurrPath = '/net/linse8-sn/home/s1222/marvin_stuff/IQA/Codes_FeatureLearning';
end
addpath(genpath(sCurrPath)); %add the path

%get patients to iterate over...
sPats = dir(sPath);%subfolders with names of patients 
lMask = cell2mat({sPats(:).isdir}); if(any(~lMask)), sPats(~lMask) = []; end
sPats = sPats(3:end);%remove . & ..


sPats(~lPats)=[];%remove patients who are shit (cb)

%load parameter set
[sPathPara,sFilenamePara] = fileparts(sParafile);
if(~isempty(sPathPara)), cd(sPathPara); end;
eval([sFilenamePara,';']);

patchSize=iPatchSize;%override patchSize with given

allPatches=[];
allY=[];
iPats=[];

for iModel=1:length(sModel)
    [sDataRef,sDataArt,~,~, sDataRef_az,sDataArt_az,...
        sDataRef_mb, sDataArt_mb,sDataRef_tk, sDataArt_tk]=fGetModelInfo(sModel{iModel});
    %% 2D/3D-Patching
    for iPat= 1:length(sPats)
        switch sPats(iPat).name
            case 'cb' %data missing
                continue;
            case 'az_old'
                if(~isempty(sDataRef_az))
                    if (strcmp(sDataRef_az, 'no_data')) continue;
                    else
                        cDataRef={sDataRef_az;0};
                        cDataArt={sDataArt_az;1}; end
                else
                    cDataRef = {sDataRef;0};
                    cDataArt = {sDataArt;1}; end
            case 'mb_old'
                if(~isempty(sDataRef_mb))
                    if (strcmp(sDataRef_mb, 'no_data')) continue;
                    else
                        cDataRef={sDataRef_mb;0};
                        cDataArt={sDataArt_mb;1}; end
                else
                    cDataRef = {sDataRef;0};
                    cDataArt = {sDataArt;1}; end
            case 'tk_old'
                if(~isempty(sDataRef_tk))
                    if (strcmp(sDataRef_tk, 'no_data'))
                        continue;
                    else
                        cDataRef={sDataRef_tk;0};
                        cDataArt={sDataArt_tk;1};
                    end
                else
                    cDataRef = {sDataRef;0};
                    cDataArt = {sDataArt;1};
                end
            otherwise
                cDataRef = {sDataRef;0};
                cDataArt = {sDataArt;1};
        end
        
        sDataAll = cat(2,cDataRef,cDataArt);
        
        for iJ= 1:size(sDataAll,2) % 0ref and 1art
            for iRot= 1:length(iRots)
                sDataIn = sDataAll{1,iJ};

                sToReadDicom=[sPath,filesep,sPats(iPat).name,filesep,sDataIn];
                dImg = fReadDICOM(sToReadDicom);

                % scaling
                dImg = scaleImg(dImg);%works for 3D too, default 0-1 range

                iImgSize = size(dImg);

                % patching
                if (l3D)%3D-case
                    dPatches = fPatch3D(dImg, patchSize, patchOverlap);
                    iCatDim=4;
                else%2D-case
                    dPatches = fPatch(dImg, patchSize, patchOverlap);
                    iCatDim=3;
                end
                dPatches=rot90(dPatches, iRots{iRot});
                allPatches = cat(iCatDim,allPatches,dPatches);
                y = sDataAll{2,iJ} .* ones(size(dPatches,iCatDim),1);
                iPats = cat(1,iPats,iPat * ones(size(dPatches,iCatDim),1));
                disp(size(allPatches))%for debug
                allY = cat(1,allY,y);
            end
        end
    end
    clearvars sDataRef_az sDataArt_az sDataRef_mb sDataArt_mb sDataRef_tk sDataArt_tk;%reset sDataArt etc...
end

if l3D
    dimensions = [size(allPatches,4), 1, size(allPatches,3), size(allPatches,1),...
        size(allPatches,2)];
else
    dimensions = [size(allPatches,3), 1, size(allPatches,1), size(allPatches,2)];
end

if strcmp(sSplitting, 'none')
    %just pass all samples in X_train/Y_train eg for crossval with scikit
    %learn or predicting
    %could be called data and labels
    [sDir,~,~]= fileparts(sOutPath);
    if(~exist(sDir,'dir')) mkdir(sDir); end
    X_train = allPatches;
    y_train = allY;
    if l3D X_train = permute(X_train,[4 5 1 2 3]);
    else X_train= permute(X_train,[3 4 1 2]); end
    
    % save for python
    save(sOutPath, 'X_train', 'y_train', 'dimensions', 'patchSize', 'patchOverlap', '-v7.3');
    %dont call python... 
elseif(strcmp(sSplitting,'normal'))
  
    if (l3D) 
        nPatches = size(allPatches,4) 
        dVal = floor(dSplitval* nPatches);
        rand_num = randperm(nPatches,dVal);
        X_test = allPatches(:,:,:,rand_num);
        X_train(:,:,:,rand_num)=[];%remove the test set
        X_train = permute(X_train,[4 5 1 2 3]);
        X_test = permute(X_test,[4 5 1 2 3]);
    else
        nPatches = size(allPatches,3)
        dVal = floor(dSplitval* nPatches);
        rand_num = randperm(nPatches,dVal);
        X_test = allPatches(:,:,rand_num);
        X_train(:,:,rand_num)=[];%remove the test set
        X_train = permute(X_train,[3 4 1 2]);
        X_test = permute(X_test,[3 4 1 2]);
    end
    y_test = allY(rand_num);

    X_train = allPatches;
    
    y_train = allY;
    y_train(rand_num)=[];

    % arrange data for usage in python and keras
    disp(size(X_train));
    disp(size(X_test));
    
    % save for python
    [sDir, sN, ext]= fileparts(sOutPath);
    if(~exist(sDir,'dir'))
            mkdir(sDir);
        end
    save(sOutPath, 'X_train', 'X_test', 'y_train', 'y_test', 'dimensions', 'patchSize', 'patchOverlap', '-v7.3');
    
   
elseif(strcmp(sSplitting,'crossvalidation_data'))%%closed
%     %kfold validation call python nFolds-Times...
%     
%     if(exist([sPathOut_3D,filesep, num2str(patchSize(1)), num2str(patchSize(2)), num2str(patchSize(3)), filesep, 'data',filesep, 'iFolds.mat'], 'file'))
%         load([sPathOut_3D,filesep, num2str(patchSize(1)), num2str(patchSize(2)), num2str(patchSize(3)), filesep, 'data',filesep, 'iFolds.mat']);
%     else
%         iInd = crossvalind('Kfold', size(allPatches,4), nFolds);
%         save([sPathOut_3D,filesep, num2str(patchSize(1)), num2str(patchSize(2)), num2str(patchSize(3)), filesep, 'data_3D',filesep, 'iFolds.mat'], 'iInd');
%     end
%     for iFold = 1:nFolds
%         if(~lPats(iFold)), continue; end;
%         fprintf('Fold %d\n', iFold);
%         lMask = iInd == iFold;
%         
%         X_test = allPatches(:,:,:,lMask);
%         y_test = allY(lMask);
%         
%         X_train = allPatches(:,:,:,~lMask);
%         y_train = allY(~lMask);
%         
%         X_train = permute(X_train,[4 5 1 2 3]);
%         X_test = permute(X_test,[4 5 1 2 3]);
%         
%         % save for python
%         sPathOutCurr = [sPathOut_3D,filesep, num2str(patchSize(1)), num2str(patchSize(2)), num2str(patchSize(3)), filesep, 'data_3D',filesep, num2str(iFold,'%02d')];
%         if(~exist(sPathOutCurr,'dir'))
%             mkdir(sPathOutCurr);
%         end
%         sFiles = dir(sPathOutCurr); if(length(sFiles(3:end)) == prod(structfun(@(x) length(x), sOptiPara))*3 + 1), continue; end % learning rates, batch sizes, 3 output files, 1 input file
% 
%         sPathOut = [sPathOutCurr,filesep,'crossVal_data_3D',num2str(iFold,'%02d'),'_', num2str(patchSize(1)), num2str(patchSize(2)), num2str(patchSize(3)),'.mat'];
%         save(sPathMat, 'X_train', 'X_test', 'y_train', 'y_test', 'dimensions', 'patchSize', 'patchOverlap', '-v7.3')
%     end
    
    
elseif(strcmp(sSplitting,'crossvalidation_patient'))
    %for iPat = 1:length(sPats)
    for iPat = 1:17
        [sDir,~,~]= fileparts(sOutPath);
        if(~exist(sDir,'dir')) mkdir(sDir); end
        
        sDir=[sDir, filesep, num2str(patchSize(1)), num2str(patchSize(2))];% add the patchSize dir
        if (l3D)  sDir=[sDir,num2str(patchSize(3))];end
        if (~exist(sDir,'dir')) mkdir(sDir); end
        if l3D
            sPathMat = [sDir,filesep,'crossVal_3D_',num2str(iPat,'%02d'),'_',...
                num2str(patchSize(1)), num2str(patchSize(2)), num2str(patchSize(3)),'.mat'];
        else
            sPathMat = [sDir,filesep,'crossVal_',num2str(iPat,'%02d'),'_',...
                num2str(patchSize(1)), num2str(patchSize(2)),'.mat'];
        end
        if (exist(sPathMat, 'file')) continue; end;%if already created, go on...
        
        if(~lPats(iPat)) continue; end;%skip missing pats
        
        lMask = (iPats == iPat);
        if(~any(lMask)) continue; end;%skip missing pats,
        %not having data for this particular model(xx_old)
        fprintf('Patient %d\n', iPat);
        if l3D
            X_test = allPatches(:,:,:,lMask);
            X_train = allPatches(:,:,:,~lMask);
            X_train = permute(X_train,[4 5 1 2 3]);
            X_test = permute(X_test,[4 5 1 2 3]);
        else
            X_test = allPatches(:,:,lMask);
            X_train = allPatches(:,:,~lMask);
            X_train = permute(X_train,[3 4 1 2]);
            X_test = permute(X_test,[3 4 1 2]);
        end
        y_test = allY(lMask);
        y_train = allY(~lMask);
        
        
        disp(size(X_train));
        disp(size(X_test));
        
        %         % save for python
        %         sPathOutCurr = [sPathOut_3D,filesep, num2str(patchSize(1)), num2str(patchSize(2)), num2str(patchSize(3)), filesep, 'patient_3D', filesep, num2str(iPat,'%02d')];
        %         sPathOutCurrPyt = [sPathOut_3D,filesep,'results',filesep, num2str(patchSize(1)), num2str(patchSize(2)), num2str(patchSize(3)), filesep, 'patient_3D', filesep, num2str(iPat,'%02d')];
        %         if(~exist(sPathOutCurr,'dir'))
        %             mkdir(sPathOutCurr);
        %         end
        %         if(~exist(sPathOutCurrPyt,'dir'))
        %             mkdir(sPathOutCurrPyt);
        %         end
        
        
        save(sPathMat, 'X_train', 'X_test', 'y_train', 'y_test', 'dimensions',...
            'patchSize', 'patchOverlap','iPat', '-v7.3');
        
    %end TODO
end

%exit;
end




