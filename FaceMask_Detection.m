w = webcam("Snap Camera");

%w.Brightness = 128;
w.Resolution = '1280x720';
viewer = vision.DeployableVideoPlayer();
viewer.Size = "Custom";
viewer.Location = [0 0];

Threshold_Value = 0.4

load('Trained Detectors\COVID19_Mask_yolo_OwnDataset.mat')
mdl = 'YOLOv2';

faceDetector = vision.CascadeObjectDetector();

while cont
    img = snapshot(w);
    sz = size(img);
    targetSize = [224 224];
    img_r = imresize(img, targetSize);
    bbox = [];
    tic;

    stream = rgb2gray(imresize(img, downSampleSize));
    boundingbox = faceDetector.step(stream);   

    [bbox, score, label] = detect(detector, img_r, 'Threshold', Threshold_Value, 'ExecutionEnvironment', "cpu");

    detectedImg = img;

    if ~isempty(bbox)
        bboxf = bboxre(bbox, sz, targetSize);
        num = numel(bboxf(:,1));

        label = reshape(label, [num,1]);
        detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', boundingbox.*PositionMultiplier, ["Masked" + " : "+string(score)], 'Color', 'green', ...
            'Fontsize', 50, 'linewidth', 8, 'textboxopacity', 1);
    else
        detectedImg = insertObjectAnnotation(detectedImg, 'rectangle', boundingbox.*PositionMultiplier, ["Unmasked "], 'Color', 'red', ...
            'Fontsize', 50, 'linewidth', 8, 'textboxopacity', 1);
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