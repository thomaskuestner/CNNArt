function fSetGPU( iGPU )
% set GPU
fid = fopen('/home/m123/.theanorc', 'r');
sConfig = fscanf(fid, '%s');
[~,iStart] = regexp(sConfig,'device=gpu');
iCurrGPU = str2double(sConfig(iStart+1));

if(exist(['/home/m123/theanorc_gpu',num2str(iGPU,'%d')], 'file'))
    fprintf('Switching GPU %d -> %d\n', iCurrGPU, iGPU);
    copyfile(['/home/m123/theanorc_gpu',num2str(iGPU,'%d')], '/home/m123/.theanorc');
else
    fprintf('Calculating on GPU %d\n', iCurrGPU);
end


end

