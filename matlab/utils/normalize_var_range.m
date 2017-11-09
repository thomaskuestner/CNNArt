function [ norm_dImg ] = normalize_var_range( dImg, norm_type , range)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% function works for 'dr' for 2D and 3D input for dImg


%kind of normalization


if exist('range') == 1
    range = range;
else
    range = 255;
end

switch norm_type
    
    case 'dr'
        
        if length(size(dImg))==2
            % normalize s.t. dynamic range
            max_val = max(max(dImg));
            min_val = min(min(dImg));
            
            dyn_range = max_val-min_val;
        else
            max_val = max(max(max(dImg)));
            min_val = min(min(min(dImg)));
            
            dyn_range = max_val-min_val;
        end
            
        % no Division by zero, if no dynamic range available
        if dyn_range == 0;
            error('No division by zero! No dynamic range.');
        else
            
            norm_dImg = bsxfun(@times, bsxfun(@minus, dImg, min_val), 1 ./ dyn_range);
            
            norm_dImg = norm_dImg*range;
        end
        
        % zero mean unit variance    
    case 'zmuv'
        
        norm_dImg = zscore(dImg);
        
        
    otherwise
        
        error('Error in fct normalize: Unknown normalization');
        
        
end

