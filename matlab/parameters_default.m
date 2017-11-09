% initialize
% scaling range
range = [0 1];
% patches
patchSize = [40 40]; % x,y
patchOverlap = 0.5; % 50%
% splitting strategy
% 'normal': percentage splitting
% 'crossvalidation_patient': cross validation on patient (leave one patient out)
% 'crossvalidation_data': cross validation on data
sSplitting = 'crossvalidation_data';
% number of folds
nFolds = 15;
% splitting in training and test set
dSplitval = 0.1;
% optimization type in keras: 'grid', 'hyperas', 'none'
sOpti = 'grid';
% optimized parameters
sOptiPara.batchSize = 128;
sOptiPara.lr = [0.1, 0.01, 0.05, 0.005, 0.001]; % -> hardcoded in *.py -> adapt