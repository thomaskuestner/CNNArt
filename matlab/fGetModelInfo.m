function [sDataRef,sDataArt,sPathOut,sOptiModel] = fGetModelInfo( sModel )
% get necessary image database information and optimal pre-trained model
% input
% sModel        desired model

% (c) Thomas Kuestner, thomas.kuestner@iss.uni-stuttgart.de, 2017

% get current path
sCurrPath = fileparts(mfilename('fullpath'));

switch sModel
    case 'motion_head'
        sDataRef = ['dicom_sorted',filesep,'t1_tse_tra_Kopf_0002'];
        sDataArt = ['dicom_sorted',filesep,'t1_tse_tra_Kopf_Motion_0003'];
        if(ispc)
            sPathOut = 'D:\med_data\MRPhysics\CNN\Headcross';
        else
            sPathOut = '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Headcross';
        end
        sOptiModel = {[sCurrPath,filesep,'bestModels',filesep,'head_3030_lr_0.0001_bs_64']; ...
                      [sCurrPath,filesep,'bestModels',filesep,'head_4040_lr_0.0001_bs_64']; ...
                      [sCurrPath,filesep,'bestModels',filesep,'head_6060_lr_0.0001_bs_64'];};
        
    case 'motion_abd'
        sDataRef = ['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_0004'];
        sDataArt = ['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_Motion_0005'];
        if(ispc)
            sPathOut = 'D:\med_data\MRPhysics\CNN\Abdcross';
        else
            sPathOut = '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Abdcross';
        end
        sOptiModel = {[sCurrPath,filesep,'bestModels',filesep,'abdomen_3030_lr_0.0001_bs_64']; ...
                      [sCurrPath,filesep,'bestModels',filesep,'abdomen_4040_lr_0.0001_bs_64']; ...
                      [sCurrPath,filesep,'bestModels',filesep,'abdomen_6060_lr_0.0001_bs_64'];};
					  
	case 'motion_all'
        sDataRef = {['dicom_sorted',filesep,'t1_tse_tra_Kopf_0002'],['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_0004']; ...
					 0												,0};
        sDataArt = {['dicom_sorted',filesep,'t1_tse_tra_Kopf_Motion_0003'],['dicom_sorted',filesep,'t1_tse_tra_fs_mbh_Leber_Motion_0005']; ...
					 1													  , 1};
        if(ispc)
            sPathOut = 'D:\med_data\MRPhysics\CNN\Allcross';
        else
            sPathOut = '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Allcross';
        end
        sOptiModel = {};
        
    case 'shim'
       sDataRef = ['dicom_sorted',filesep,'t2_tse_tra_fs_Becken_0009'];
       sDataArt = ['dicom_sorted',filesep,'t2_tse_tra_fs_Becken_Shim_xz_0012'];
       if(ispc)
           sPathOut = 'D:\med_data\MRPhysics\CNN\Shimcross';
       else
           sPathOut = '/med_data/ImageSimilarity/Databases/MRPhysics/CNN/Shimcross';
       end
       sOptiModel = [sCurrPath,filesep,'bestModels',filesep,'MISSING'];
        
    case 'noise'
        
end

end

