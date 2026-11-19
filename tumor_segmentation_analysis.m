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



function printScores(methodName, fileName, pred, enhancing, wholeTumor)
    fprintf("%s | %s | Dice vs enhancing: %.3f | Jaccard vs Enchancing: %.3f | Dice vs whole: %.3f\n", ...
        methodName, fileName, ...
        dice(pred, enhancing), ...
        jaccard(pred, enhancing), dice(pred, wholeTumor));
end
                                                            %SegMethod is a
                                                            %ANON function
function predictions = scoreSliceImages(methodName, segmentMethod, testMatchingFiles, enhancing, wholeTumor)
    predictions = cell(size(testMatchingFiles));

    for i = 2:numel(testMatchingFiles)
        file = testMatchingFiles(i);
        img = imread(fullfile(file.folder, file.name));

        pred = segmentMethod(img);
        predictions{i} = pred;
        printScores(methodName, file.name, pred, enhancing, wholeTumor);
    end
    fprintf("\n")
end

function printCompTitles(mode, img, wholeTumor, otsuPred, multiOtsuPred)
    nexttile;
    imshow(img, []);
    title(upper(mode) + "MRI");
    
    nexttile;
    imshow(wholeTumor);
    title("Reference: Whole Tumor");
    
    nexttile;
    imshow(otsuPred);
    title("Otsu");
    
    nexttile;
    imshow(multiOtsuPred);
    title("Multi-Otsu");
end

function showSliceComparison(testMatchingFiles, wholeTumor, ostuMasks, multiOtsuMasks)
    close all;
    % T1C: T1-wted with constrast: brightens areas, enhancing tumor areas
    % T1N: T1-wted with no constrast, Brain Anatomy with pre-contrast
    % appearance
    
    % T2W: T2-wted Water rich areas, including edema and cerebrospinal fluid  
    
    fileNames = string({testMatchingFiles.name});
    
    for mode = ["t1c", "t2w"]
        suffix = "-" + mode + "_slice_053.png";
        % idx = find(endswith(string({fileNames}), suffix), 1);
        % idx = find(contains(string({fileNames}), suffix), 1);
        idx = find(contains(fileNames, suffix), 1);
    
        if isempty(idx)
            warning("No matching PNG found for %s.", mode);
            continue
        end
    
        file = testMatchingFiles(idx);
        img = imread(fullfile(file.folder, file.name));
    
        otsuPred = ostuMasks{idx};
        multiOtsuPred = multiOtsuMasks{idx};
    
        figure("Name", file.name, "NumberTitle", "off");
        tiledlayout(1, 4, "TileSpacing", "Compact");
    
        printCompTitles(mode, img, wholeTumor, otsuPred, multiOtsuPred)
    end
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
    
    otsuMasks = scoreSliceImages("Otsu", @OtsuMethod, testMatchingFiles, enhancing, wholeTumor);
    multiOtsuMasks = scoreSliceImages("Multi-Otsu", @MultiOtsuMethod, testMatchingFiles, enhancing, wholeTumor);

    showSliceComparison(testMatchingFiles, wholeTumor, otsuMasks, multiOtsuMasks);

    % This was moved to the generic function for both Otsu and MultiOtsu
        % You can pass in an ANON function to run the appropriate function
        % on the data
    % for i = 2:length(testMatchingFiles)
    % 
    %     % Step 1: Brain-only mask
    % 
    %     img = imread(fullfile( ...
    %                  testMatchingFiles(i).folder, ...
    %                  testMatchingFiles(i).name));
    % 
    %     % Step 2: Otsu's method
    %     otsu_pred = OtsuMethod(img);
    %     otsu_similarity = dice(otsu_pred, groundTruth);
    % 
    %     fileName = testMatchingFiles(i).name;
    %     printScores("Otsu", fileName, otsu_pred, enchancing, wholeTumor);
    % 
    %     % Step 3: Multi-Otsu
    % 
    %     multi_otsu_pred = MultiOtsuMethod(img);
    %     printScores("Multi Otsu", fileName, multi_otsu_pred, enchancing, wholeTumor);
    % end
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

                                % Lol JAN
function path = deterPath(name, Joesph_sharedPath, Aaron_sharedPath, Neeka_sharedPath)
    if(name == "Joesph")
        path = Joesph_sharedPath;
    elseif(name == "Aaron")
          path = Aaron_sharedPath;
    elseif(name == "Neeka")
        path = Neeka_sharedPath;
    else
        path = "INVALID";
    end
end

% Loading in data, for now replace the below with your own local files
Joesph_sharedPath = "/Users/josephrexrode/UTDallas/2026-2027/MATH6346/FinalProject";
Aaron_sharedPath = "C:/Users/aaron/Downloads/School/UTD/Medical Image Analysis/Final Project/";
Neeka_sharedPath = "UNKNOWN"; % Neeka please add lol

name = "Aaron";
sharedPath = deterPath(name, Joesph_sharedPath, Aaron_sharedPath, Neeka_sharedPath);

% Extract file paths
parentFolder  = fullfile(sharedPath,   "BraTS2026subset");
outputFolder  = fullfile(sharedPath,   "2DSlices");
rawFolder     = fullfile(outputFolder, "raw");
segmentFolder = fullfile(outputFolder, "groundTruth");

% Run slicing algos: (Multi)Otsu
niiFiles = loadFiles(parentFolder, outputFolder, rawFolder, segmentFolder);
convert2DFiles(niiFiles, rawFolder, segmentFolder);
runSliceTest(outputFolder);

