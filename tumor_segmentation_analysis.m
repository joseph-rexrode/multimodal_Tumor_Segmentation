function niiFiles = loadFiles(parentFolder, outputFolder, rawFolder, segmentFolder)
    % creates directory if it doesn't already exist
    if ~exist(outputFolder, "dir")
        mkdir(outputFolder);
        mkdir(segmentFolder);
        mkdir(rawFolder);
    end
    
    % Listing 3D files
    niiFiles = dir(fullfile(parentFolder, '**', '*.nii.gz'));
end 

function convert2DFiles(niiFiles, rawFolder, segmentFolder)
    % Converting to 2D files - this creates a lot of files but they're small dw
    for i = 1:length(niiFiles)
        filePath = fullfile(niiFiles(i).folder, niiFiles(i).name);
    
        vol = niftiread(filePath);
    
        [~, baseName, ~] = fileparts(niiFiles(i).name);
        [~, baseName, ~] = fileparts(baseName);
    
        [~, subFolderName] = fileparts(niiFiles(i).folder);
    
        numSlices = size(vol, 3);
    
        for j = 1:numSlices
            slice2D = vol(:, :, j);
    
            sliceNorm = mat2gray(slice2D);
    
                outName = sprintf('%s_%s_slice_%03d.png', subFolderName, baseName, j);
    
            % creating separate folder for the ground truth segmentation
            if baseName(length(baseName)-2:length(baseName)) == "seg"
                imwrite(sliceNorm, fullfile(segmentFolder, outName))
            else
                imwrite(sliceNorm, fullfile(rawFolder, outName));
            end
        end
    end
end


%% 
% testing median filter first (tried this with like 10 slices earlier,
% DONT RUN THIS PART UNLESS YOU WANT A BAJILLION FIGURES)


% rawFiles = dir(fullfile(rawFolder, "*.png"));
% 
% medianFolder = "/Users/josephrexrode/UTDallas/2026-2027/MATH6346/FinalProject/2DSlices/median";
% 
% if ~exist(medianFolder, "dir")
%     mkdir(medianFolder);
% end
% 
% for i = 1:length(rawFiles)
%     rawimgname = fullfile(rawFiles(i).folder, rawFiles(i).name);
%     rawimg = imread(rawimgname);
% 
%     figure;
%     tiledlayout(1, 3);
%     nexttile();
%     imshow(rawimg);
%     title("Raw:");
%     nexttile();
%     imshow(medfilt2(rawimg));
%     title("Median Filter:");
%     nexttile();
%     imshow()
% end

% didn't look too different but keeping it here for posterity

%%

% testing one slice: 8-53

function pred = OtsuMethod(img)
    brainOnly = img > 0;
    threshold = graythresh(img(brainOnly));
    pred = imbinarize(img, threshold) & brainOnly;
end


function pred = MultiOtsuMethod(img)
    brainOnly = img > 0;
    threshold = multithresh(img(brainOnly), 2);
    pred = (img > threshold(end)) & brainOnly;
end

function runSliceTest(outputFolder)
    testFilePattern = fullfile(outputFolder, ...
                        '*', ...
                        '*00008*slice_053.png');
    testMatchingFiles = dir(testFilePattern);
    
    % BraTS segmentation being used as ground truth
    groundTruth = imread(fullfile( ...
                            testMatchingFiles(1).folder, ...
                            testMatchingFiles(1).name));
    
    % categories for different parts of tumor, to be used for various
    % modalities as some reveal parts better than others
    
    wholeTumor = groundTruth > 0;
    enhancing = groundTruth == 255;
    tumorCore = (groundTruth == 85) | (groundTruth == 255);
    
    for i = 2:length(testMatchingFiles)
    
        % Step 1: Brain-only mask
    
        img = imread(fullfile( ...
                     testMatchingFiles(i).folder, ...
                     testMatchingFiles(i).name));
    
        % Step 2: Otsu's method
        otsu_pred = OtsuMethod(img)
        otsu_similarity = dice(otsu_pred, groundTruth);
    
        fprintf('%s | Dice vs enhancing: %.3f | Jaccard: %.3f | Dice vs whole: %.3f\n', ...
            testMatchingFiles(i).name, dice(otsu_pred, enhancing), ...
            jaccard(otsu_pred, enhancing), dice(otsu_pred, wholeTumor));
    
        % Step 3: Multi-Otsu

        multi_otsu_pred = MultiOtsuMethod(img)
    
    
        
    end
end


%% Evaluation Functions

function roll = dice(pred, gt)
    roll = 2*nnz(pred & gt) / (nnz(pred) + nnz(gt));
end

function score = jaccard(pred, gt)
    score = nnz(pred & gt) / nnz(pred | gt);
end

% main % main % main % main % main % main % main % main % main % main % main % main % main  
%% 


% Loading in data, for now replace the below with your own local files

sharedPath = "/Users/josephrexrode/UTDallas/2026-2027/MATH6346/FinalProject";

parentFolder  = strcat(sharedPath, "/BraTS2026subset");
outputFolder  = strcat(sharedPath, "/2DSlices");
rawFolder     = strcat(sharedPath, "/2DSlices/raw");
segmentFolder = strcat(sharedPath, "/2DSlices/groundTruth");

niiFiles = loadFiles(parentFolder, outputFolder, rawFolder, segmentFolder);
convert2DFiles(niiFiles, rawFolder, segmentFolder);
runSliceTest(outputFolder);