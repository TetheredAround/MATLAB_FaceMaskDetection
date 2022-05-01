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
        path = path + "" + file;
end

load(path)

obj_indx = questdlg('What to Detect?', ...
	'Object Detection', 'Head', 'Mask', 'Head');

[file,path] = uigetfile('*.png');
img_file = path + "" + file;

mdl = 'YOLOv2';

faceDetector = vision.CascadeObjectDetector();

downSampleSize = 1;
PositionMultiplier = 1/downSampleSize;

img = imread(img_file);

cont = 1;

while cont
    sz = size(img);
    targetSize = [(sz(2)*downSampleSize) (sz(1)*downSampleSize)];
    img_r = imresize(img, targetSize);
    bbox = [];
    tic;

    stream = im2gray(imresize(img, downSampleSize));
    boundingbox = faceDetector.step(stream);

    [bbox, score, label] = detect(detector, img_r, 'Threshold', 0.8, 'ExecutionEnvironment', "cpu");

    detectedImg = img;

    if ~isempty(bbox)
        bboxf = bboxre(bbox, sz, targetSize);
        num = numel(bboxf(:,1));

        label = reshape(label, [num,1]);
        if obj_indx == 'Head'
            detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', boundingbox.*PositionMultiplier, ["Masked"], 'Color', 'green', ...
                'Fontsize', 50, 'linewidth', 8, 'textboxopacity', 1);
        else
            detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', bboxf, [string(label)+ " : "+string(score)], 'Color', 'green', ...
               'Fontsize', 50, 'linewidth', 8, 'textboxopacity', 1);
        end
        detectedImg = insertText(detectedImg, [Center_X, 1],  "      Face & Mask Detected!       ", 'FontSize', 35, 'BoxColor', 'g');
    elseif isempty(bbox) && ~isempty(boundingbox)
        detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', boundingbox.*PositionMultiplier, ["Unmasked "], 'Color', 'red', ...
            'Fontsize', 50, 'linewidth', 8, 'textboxopacity', 1);
        detectedImg = insertText(detectedImg, [Center_X, 1],  "      No Mask Detected!      ", 'FontSize', 35, 'BoxColor', 'r');
    else
        detectedImg = insertText(detectedImg, [Center_X, 1],  "      No Face Detected!      ", 'FontSize', 35, 'BoxColor', [0.85 0.85 0.85]);
    end
    
    imshow(detectedImg)
end
release(imshow)

function bbox = bboxre(bbox, sz, targetSize)
bbox(:,1) = bbox(:,1)*sz(2)/targetSize(2);
bbox(:,2) = bbox(:,2)*sz(1)/targetSize(1);
bbox(:,3) = bbox(:,3)*sz(2)/targetSize(2);
bbox(:,4) = bbox(:,4)*sz(1)/targetSize(1);
end