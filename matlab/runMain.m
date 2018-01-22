%% train network
iGPU = 0;
lPats = true(1,15); lPats(3) = false;
sParafile = 'parameters_default.m'; sModel = 'motion_head';
fTrainCNN(sParafile, sModel, lPats, iGPU);


%% predict
iGPU = 1;
sParafile = 'para_crossvalPat.m'; sModel = 'motion_head';
sDICOMPath = '/med_data/ImageSimilarity/Databases/MRPhysics/newProtocol/az_old/dicom_sorted/t1_tse_tra_Kopf_0002';
fPredictCNN( sParafile, sDICOMPath, sModel, iGPU )


%% visualize overlay
iPat = 2;
sParafile = 'para_crossvalPat'; sModel = 'motion_head';
sType = 'art'; % 'art' | 'ref'
sPath = '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Headcross/4040';
load([sPath,filesep,'patient',filesep,num2str('%02',iPat),filesep,sprintf('outcrossVal%02d4040_lr_0.0001_bs_64.mat',iPat)]);
if(strcmp(sType,'ref')) % first read in as test data
    dProbPatch = prob_test(1:end/2,:);
elseif(strcmp(sType,'art'))
    dProbPatch = prob_test(end/2+1:end,:);
end
[ hfig, dImg, dProbOverlay ] = fVisualizeOverlay( sParafile, sModel, iPat, sType, dProbPatch );


%% visualize significant points
iPats = [2,6];
sParafile = 'para_crossvalPat'; sModel = 'motion_head';
[dDeepVis, dSubsets] = fVisualizePoint( iPats, sModel, sParafile );
