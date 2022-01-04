[temp_indx,temp_tf] = listdlg('PromptString',{'Select camera to be used.',...
    'Only one camera can be used.',''},...
    'SelectionMode','single','ListString', webcamlist());

w = webcam(temp_indx);

[temp_indx,temp_tf] = listdlg('PromptString',{'Select resolution to be used for the camera.',...
    'Only one resolution can be used.',''},...
    'SelectionMode','single','ListString', w.AvailableResolutions());

%w.Brightness = 128;
w.Resolution = w.AvailableResolutions{temp_indx};
viewer = vision.DeployableVideoPlayer();
viewer.Size = "Custom";
viewer.Location = [0 0];

options = {"Researchers' dataset", 'Original YOLOv2', 'Masked Face Net (1000)', 'Masked Face Net (150)', 'Kaggle', 'Custom'};

[temp_indx,temp_tf] = listdlg('PromptString',{'Select a detector model.',...
    'Only one file can be selected.',''},...
    'SelectionMode','single','ListString',options);

switch temp_indx
    case 1
        path = 'Trained Detectors\COVID19_Mask_yolo_OwnDataset.mat';
    case 2
        path = 'Trained Detectors\COVID19_Mask_yolo_Original.mat';
    case 3
        path = 'Trained Detectors\COVID19_Mask_yolo_MaskedFaceNet1000.mat';
    case 4
        path = 'Trained Detectors\COVID19_Mask_yolo_MaskedFaceNet150.mat';
    case 5
        path = 'Trained Detectors\COVID19_Mask_yolo_Kaggle.mat';
    case 6
        [file,path] = uigetfile('*.mat');
end

load(path);
mdl = 'YOLOv2';

faceDetector = vision.CascadeObjectDetector();

downSampleSize = 0.5;
PositionMultiplier = 1/downSampleSize;

cont = 1;

while cont
    img = snapshot(w);
    sz = size(img);
    %targetSize = [224 224];
    targetSize = [(sz(1)*downSampleSize) (sz(2)*downSampleSize)];
    img_r = imresize(img, targetSize);
    bbox = [];
    tic;

    stream = rgb2gray(imresize(img, downSampleSize));
    boundingbox = faceDetector.step(stream);
    
    %First Gen Model shows below above 20% (COVID19_Mask_Yolo.mat)
    %Second Gen Model shows below 10%
    %Third Gen Model shows simalar results if not worse than second gen

    [bbox, score, label] = detect(detector, img_r, 'Threshold', 0.8, 'ExecutionEnvironment', "cpu");

    detectedImg = img;

    if ~isempty(bbox)
        bboxf = bboxre(bbox, sz, targetSize);
        num = numel(bboxf(:,1));

        label = reshape(label, [num,1]);
        detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', boundingbox.*PositionMultiplier, ["Masked"], 'Color', 'magenta', ...
           'Fontsize', 50, 'linewidth', 8, 'textboxopacity', 1);
%       detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', bboxf, [string(label)+ " : "+string(score)], 'Color', 'magenta', ...
%           'Fontsize', 50, 'linewidth', 8, 'textboxopacity', 1);
        detectedImg = insertText(detectedImg, [600, 1],  "      Mask Detected!       ", 'FontSize', 35, 'BoxColor', 'g');
    else
        detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', boundingbox.*PositionMultiplier, ["Unmasked "], 'Color', 'magenta', ...
            'Fontsize', 50, 'linewidth', 8, 'textboxopacity', 1);
        detectedImg = insertText(detectedImg, [600, 1],  "      No Mask Detected!      ", 'FontSize', 35, 'BoxColor', 'r');   
    end
    
    viewer(detectedImg)
    cont = isOpen(viewer);
end
release(viewer)

function bbox = bboxre(bbox, sz, targetSize)
bbox(:,1) = bbox(:,1)*sz(2)/targetSize(2);
bbox(:,2) = bbox(:,2)*sz(1)/targetSize(1);
bbox(:,3) = bbox(:,3)*sz(2)/targetSize(2);
bbox(:,4) = bbox(:,4)*sz(1)/targetSize(1);
end