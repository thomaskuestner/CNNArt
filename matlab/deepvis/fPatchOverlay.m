function hfig = fPatchOverlay( dImg, dPatch, iScale, dAlpha, sPathOut, cPlotLimits, lLabel, lGray)
%FPATCHOVERLAY overlay figure   
    
    if(nargin < 8)
        lGray = false;
    end
    if(nargin < 7)
        lLabel = true;
    end
    if(nargin < 6)
        xLimits = [1 size(dImg,2)];
        yLimits = [1 size(dImg,1)];
    else
        xLimits = cPlotLimits{1};
        yLimits = cPlotLimits{2};
    end
    if(nargin < 5)
        sPathOut = cd;
    end
    if(nargin < 4)
        dAlpha = 0.6;
    end
    if(nargin < 3)
        iScale = [0 1; 0 1];
    end

    h.sPathOut = sPathOut;
    h.dAlpha = dAlpha;
    h.lGray = lGray;
    h.colRange = iScale;

	hfig = figure;
    
    dImg = ((dImg - min(dImg(:))).*(h.colRange(1,2)-h.colRange(1,1)))./(max(dImg(:) - min(dImg(:))));
    if(h.lGray)        
        alpha = bsxfun(@times, ones(size(dPatch,1), size(dPatch,2)), .6);

        % find a scale dynamically with some limit
        Foreground_min = min( min(dPatch(:)), h.colRange(1) );
        Foreground_max = max( max(dPatch(:)), h.colRange(2) );
        Background_blending = bsxfun(@times, dImg, bsxfun(@minus,1,alpha));
        Foreground_blending = bsxfun( @times, bsxfun( @rdivide, ...
            bsxfun(@minus, dPatch, Foreground_min), ... 
            Foreground_max-Foreground_min ), alpha );
        h.dImg = Background_blending + Foreground_blending;
        h.hI = imshow(h.dImg(:,:,1), h.colRange); 
    else
        h.hI = axes();
        h.dImg = dImg;
        h.dPatch = dPatch;
        [h.hFront,h.hBack] = imoverlay(dImg(:,:,1,1),dPatch(:,:,1,1),h.colRange(1,:),h.colRange(2,:),'jet',h.dAlpha, h.hI);
    end
    
    xlim(xLimits);
    ylim(yLimits);
    h.hA = gca;
    h.iActive = 1;
    if(lLabel)
%         h.hT = uicontrol('Style','text', 'units','normalized', 'Position', [0.925 0.975 0.075 0.0255],'String',sprintf('I: [%.2f:%.2f]', h.colRange(1), h.colRange(2)),'ForegroundColor','k','Backgroundcolor','w');
        h.hT = uicontrol('Style','text', 'units','normalized', 'Position', [0.925 0.975 0.075 0.0255],'String',sprintf('%02d/%02d', h.iActive, size(h.dImg,3)),'ForegroundColor','k','Backgroundcolor','w');
    end
    h.lLabel = lLabel;
    set(h.hA, 'Position', [0 0 1 1]);
    set(hfig, 'Position', [0 0 size(dImg, 2).*4 size(dImg, 1).*4]);
    set(hfig, 'WindowScrollWheelFcn', @fScroll);
    set(hfig, 'KeyPressFcn'         , @fKeyPressFcn);
    currpath = fileparts(mfilename('fullpath'));
    addpath(genpath([fileparts(fileparts(currpath)),filesep,'export_fig']));
%     set(hfig, 'WindowButtonDownFcn', @fWindowButtonDownFcn);
%     set(hfig, 'WindowButtonMotionFcn', @fWindowButtonMotionFcn);
%     set(hfig, 'WindowButtonUpFcn', @fWindowButtonUpFcn);
    movegui('center');
    hold on
    
    guidata(hfig, h);
end

function fScroll(hObject, eventdata, handles)

    h = guidata(hObject);
    if eventdata.VerticalScrollCount < 0
        h.iActive = max([1 h.iActive - 1]);
    else
        h.iActive = min([size(h.dImg, 3) h.iActive + 1]);
    end

    if(h.lGray)
        set(h.hI, 'CData', h.dImg(:,:,h.iActive));
    else
        [h.hFront,h.hBack] = imoverlay(h.dImg(:,:,h.iActive,1),h.dPatch(:,:,h.iActive,1),h.colRange(1,:),h.colRange(2,:),'jet',h.dAlpha, h.hI);
    end
    set(h.hT, 'String', sprintf('%02d/%02d', h.iActive, size(h.dImg,3)));
    
    guidata(hObject, h);
end

function fKeyPressFcn(hObject, eventdata)
    if(strcmpi(eventdata.Key,'p'))
        h = guidata(hObject);
        set(h.hT, 'Visible', 'off');
        if(~exist(h.sPathOut,'dir'))
            mkdir(h.sPathOut);
        end
        sFiles = dir(h.sPathOut);
        iFound = cellfun(@(x) ~isempty(x), regexp({sFiles(:).name},[num2str(h.iActive,'%03d')]));
        if(any(iFound))
            sFile = [num2str(h.iActive,'%03d'),'_',nnz(iFound)];
        else
            sFile = num2str(h.iActive,'%03d');
        end
%         iFile = nnz(~cell2mat({sFiles(:).isdir})) + 1;
%         sFile = num2str(iFile);
        try
            export_fig([h.sPathOut,filesep,sFile,'.tif']);
        catch
            warning('export_fig() not on path');
        end
        set(h.hT, 'Visible', 'on');
    end
    guidata(hObject, h);
end

