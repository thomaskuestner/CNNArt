function fPredictCNN( sParafile, sDICOMPath, sModel, iGPU, sPathOutIn, model_nameIn )
%% test/predict an image
% input
% sParafile     parameter file
% sDICOMPath    to be predicted image
% sModel        used CNN model architecture
% iGPU          used GPU

% (c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de, 2017

% get current path
sCurrPath = fileparts(mfilename('fullpath'));
if(isempty(sCurrPath))
    sCurrPath = '/net/linse8-sn/home/m123/Matlab/ArtefactClassification';
end
addpath(genpath(sCurrPath));

% load parameter set
[sPathPara,sFilenamePara] = fileparts(sParafile);
if(~isempty(sPathPara)), cd(sPathPara); end;
eval([sFilenamePara,';']);
if(~isempty(sPathPara)), cd(sCurrPath); end;

% define optimal models
if(nargin < 5)
	[~,~,sPathOut,model_name] = fGetModelInfo( sModel );
else
	sPathOut = sPathOutIn;
	model_name = model_nameIn; % HARD-CODED!!!
end

imgPred = fReadDICOM(sDICOMPath);
imgPred = scaleImg(imgPred,iScaleRange);
dPatches = fPatch(imgPred, patchSize, patchOverlap);
X_test =  permute(dPatches,[3 4 1 2]);
X_train = 0;
y_train = 0;
y_test = ones(size(X_test,1),1);
save([sPathOut,filesep,'img_pred.mat'], 'X_test', 'model_name', 'X_train', 'y_train', 'y_test', 'patchSize');

% call python
fSetGPU( iGPU );
system(sprintf('python2 cnn_main.py -i %s -o %s -m %s', [sPathOut,filesep,'img_pred.mat'], sPathOut, sModel));

end

