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

function pred = MultiOtsuMethod(img, threshold)
    brainOnly = img > 0;
    threshold = multithresh(img(brainOnly), 2);
    pred = (img > threshold(end)) & brainOnly;
end

function pred = KMeansMethod(img, k)
    if nargin < 2, k = 4; end
    brainOnly = img > 0;
    x = double(img(brainOnly));

    rng(1);                                    % reproducible clusters
    [idx, C] = kmeans(x, k, 'Replicates', 5, 'MaxIter', 200);

    [~, tumorCluster] = max(C);                % brightest centroid = tumor candidate
    pred = false(size(img));
    pred(brainOnly) = (idx == tumorCluster);
end

function pred = KMeansMultiMethod(stack, k, targetChannel)
    brainOnly = any(stack > 0, 3);
    nCh = size(stack, 3);
    
    feat = reshape(double(stack), [], nCh);
    feat = zscore(feat(brainOnly(:), :));      % put modalities on a common scale
    
    rng(1);
    [idx, C] = kmeans(feat, k, 'Replicates', 5, 'MaxIter', 200);
    
    [~, tumorCluster] = max(C(:, targetChannel));
    pred = false(size(brainOnly));
    pred(brainOnly) = (idx == tumorCluster);
end

function pred = FuzzyCMeansMethod(img, k)
    brainOnly = img > 0;
    x = double(img(brainOnly));
    
    [centers, U] = fcm(x, k);
    [~, tumorCluster] = max(centers);
    [~, labels] = max(U, [], 1);

    pred = false(size(img));
    pred(brainOnly) = labels == tumorCluster;
end

function printScores(methodName, fileName, pred, enhancing, wholeTumor)
    fprintf("%s | %s | Dice vs Enhancing: %.3f | Jaccard vs Enchancing: %.3f | Dice vs Whole: %.3f\n", ...
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

% Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints % Prints 
function printMRIType(mode, img)
    nexttile;
    imshow(img, []);
    title(upper(mode) + "MRI");
end

function printRef(img)
    nexttile;
    imshow(img);
    title("Reference: Whole Tumor");
end

function printMultiOtsu(multiOtsuPred)
    nexttile;
    imshow(multiOtsuPred);
    title("Multi-Otsu");
end

function printOtsu(otsuPred)
    nexttile;
    imshow(otsuPred);
    title("Otsu");
end

function printMultimodalMask(multimodalMask, displayK)
    nexttile;
    imshow(multimodalMask);
    title(sprintf("Multimodal K-means (k=%d)", displayK));
end

function printKMeans(fuzzyPred)
    nexttile;
    imshow(fuzzyPred);
    title("Fuzzy C-Means (k=4)");
end


function printCompTitles(mode, img, wholeTumor, otsuPred, multiOtsuPred, multimodalMask, displayK, fuzzyPred)
    printMRIType(mode, img);
    printRef(wholeTumor);
    printOtsu(otsuPred);
    printMultiOtsu(multiOtsuPred);
    printMultimodalMask(multimodalMask, displayK);
    printKMeans(fuzzyPred);
end

% Prints % Prints % Prints % Prints % Prints % Prints 

function showSliceComparison(testMatchingFiles, wholeTumor, ostuMasks, multiOtsuMasks, displayMasks, displayK, CfuzzyMasks)
    close all;
    % T1C: T1-wted with constrast: brightens areas, enhancing tumor areas
    % T1N: T1-wted with no constrast, Brain Anatomy with pre-contrast
    % appearance
    
    % T2W: T2-wted Water rich areas, including edema and cerebrospinal fluid  
    
    fileNames = string({testMatchingFiles.name});
    
    for mode = ["t1c", "t2w", "t2f", "t1n"]
        suffix = "-" + mode + "_slice_053.png";
        % Ideally endswith should've worked but I guess I need to update my
        % MATLAB? -Aaron
        % idx = find(endswith(string({fileNames}), suffix), 1);
        % idx = find(contains(string({fileNames}), suffix), 1);
        idx = find(contains(fileNames, suffix), 1);
        multimodalMask = displayMasks{idx};
    
        if isempty(idx)
            warning("No matching PNG found for %s.", mode);
            continue
        end
    
        file = testMatchingFiles(idx);
        img = imread(fullfile(file.folder, file.name));
    
        otsuPred = ostuMasks{idx};
        multiOtsuPred = multiOtsuMasks{idx};
        fuzzyPred = CfuzzyMasks{idx};
    
        figure("Name", file.name, "NumberTitle", "off");
        tiledlayout(1, 6, "TileSpacing", "Compact");
    
        printCompTitles(mode, img, wholeTumor, otsuPred, ...
        multiOtsuPred, multimodalMask, displayK, fuzzyPred)
    end
end

function kMeansMasks = runKmeans(testMatchingFiles, enhancing, wholeTumor)
    kMeansMasks = cell(1, 5);

    for k = 2:5
        kMeansMasks{k} = scoreSliceImages(sprintf("KMeans k=%d", k), ...
            @(img) KMeansMethod(img, k), testMatchingFiles, enhancing, wholeTumor);
    end
end

function runSliceTest(outputFolder, method)
    arguments
       outputFolder,
       method ="All"
    end
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
    
    if(method == "Otsu" || method == "All")
        otsuMasks = scoreSliceImages("Otsu", @OtsuMethod, testMatchingFiles, enhancing, wholeTumor);
    end
    if(method == "Multi-Otsu" || method == "All")
        multiOtsuMasks = scoreSliceImages("Multi-Otsu", @MultiOtsuMethod, testMatchingFiles, enhancing, wholeTumor);
    end
    if(method == "K-means" || method == "All")
        kMeansMasks = runKmeans(testMatchingFiles, enhancing, wholeTumor);
    end
    if(method == "Fuzzy C-Means" || method == "All")
        CfuzzyMasks =  scoreSliceImages("Fuzzy C-Means", ...
        @(img) FuzzyCMeansMethod(img, 4), ...
        testMatchingFiles, enhancing, wholeTumor);
    end
    displayK = 4;
    displayMasks = kMeansMasks{displayK};
    
    showSliceComparison(testMatchingFiles, wholeTumor, ...
        otsuMasks, multiOtsuMasks, ...
        displayMasks, displayK, CfuzzyMasks);
end
%% Evaluation / Helper Functions

function [stack, gt] = loadSliceStack(outputFolder, patientID, sliceNum)
    files = dir(fullfile(outputFolder, '*', ...
        sprintf('*%s*slice_%03d.png', patientID, sliceNum)));
    names = {files.name};
    
    mods = ["t1n" "t1c" "t2w" "t2f"];
    imgs = cell(1, numel(mods));
    for m = 1:numel(mods)
        f = files(contains(names, "-" + mods(m) + "_"));
        imgs{m} = imread(fullfile(f(1).folder, f(1).name));
    end
    stack = cat(3, imgs{:});
    
    f = files(contains(names, "-seg_"));
    gt = imread(fullfile(f(1).folder, f(1).name));

    size(stack)
    size(groundTruth)
    unique(groundTruth)
end

function [inputImage, groundTruth, wholeTumor] = preprocessingSlice(PNGfiles)
    fileNames = string({PNGfiles.name});

    modes = ["t1n", "t1c", "t2w", "t2f"];

    firstIdx   = find(contains(fileNames, "-" + modes(1) + "_"), 1);
    % disp(firstIdx)
    firstImage = imread(fullfile(PNGfiles(firstIdx).folder, PNGfiles(firstIdx).name));
    inputImage = zeros([size(firstImage), 4], "like", firstImage);

    for c = 1:numel(modes)
        idx = find(contains(fileNames, "-" + modes(c) + "_"), 1);
        inputImage(:, :, c) = imread(fullfile(PNGfiles(idx).folder, PNGfiles(idx).name));
    end

    segIdx      = find(contains(fileNames, "-seg_"), 1);
    groundTruth = imread(fullfile(PNGfiles(segIdx).folder, PNGfiles(segIdx).name));
    wholeTumor  = groundTruth > 0;

    % disp(size(inputImage));
    % disp(size(wholeTumor));
end

function [segFiles, segNames, patientIDs] = extractSegInfo(segmentFolder)
    segFiles = dir(fullfile(segmentFolder, "*-seg_slice_*.png"));
    % numel(segFiles);
    segNames = string({segFiles.name});
    fprintf("Total labeled slices: %d\n", numel(segFiles));
end

function patientIDs = extractPatientInfo(segNames)
    patientIDs = unique(extractBetween(segNames, "BraTS-MET-", "-000_"));
    for p = patientIDs
        patientMask = contains(segNames, "BraTS-MET-" + p + "-");
        fprintf("Patient %s: %d labeled slices\n", p, nnz(patientMask));
    end
end

function [trainIdx, validationIdx, testIdx] = extractIndices(segNames, trainPatients, testPatient, validationPatient)
    patientForSlice = extractBetween(segNames, "BraTS-MET-", "-000_");

    trainIdx      = ismember(patientForSlice, trainPatients);
    validationIdx = patientForSlice == validationPatient;
    testIdx       = patientForSlice == testPatient;
    
    fprintf("Training Slices: %d\n",   nnz(trainIdx));
    fprintf("Validation Slices: %d\n", nnz(validationIdx));
    fprintf("Test Slices: %d\n",       nnz(testIdx));
end

function [inputImages, wholeTumorMasks, trainIdx, validationIdx, testIdx, segFiles] = ...
    patientTrainingDataRun(segmentFolder, outputFolder)
    [segFiles, segNames] = extractSegInfo(segmentFolder);
    patientIDs = extractPatientInfo(segNames);
    
    % Training Data 
    % 775 samples split by patient
    trainPatients = ["00002", "00004", "00006"];
    validationPatient = "00008";
    testPatient = "00009";
        
    [trainIdx, validationIdx, testIdx] = ... 
        extractIndices(segNames, trainPatients, testPatient, validationPatient);
    
    sF = numel(segFiles);
    inputImages = zeros(240, 240, 4, sF, "uint8");
    wholeTumorMasks = false(240, 240, sF);

    allPNGfiles = dir(fullfile(outputFolder, "*", "*slice_*.png"));
    allPNGnames = string({allPNGfiles.name});
    
    for i = 1:numel(segFiles)
        fileName  = string(segFiles(i).name);
        patientID = extractBetween(fileName, "BraTS-MET-", "-000_");
        sliceNum  = str2double(extractBetween(fileName, "_slice_", ".png"));

        sliceText = extractBetween(fileName, "_slice_", ".png");
        
        sliceFiles = allPNGfiles( ...
            contains(allPNGnames, "BraTS-MET-" + patientID + "-") & ...
            endsWith(allPNGnames, "_slice_" + sliceText + ".png"));

        [inputImage, groundTruth, wholeTumor] = preprocessingSlice(sliceFiles);
        inputImages(:, :, :, i) = inputImage;
        wholeTumorMasks(:, :, i) = wholeTumor;
    
        if mod(i, 25) == 0 || i == numel(segFiles)
            fprintf("Loaded %d of %d slices\n", i, numel(segFiles));
        end
    end    
end

function [inputImages, wholeTumorMasks, ...
    trainIdx, validationIdx, testIdx, segFiles]...
    = patientTrainingData(segmentFolder, outputFolder)

    cacheFile = fullfile(outputFolder, "trainingData.mat");
    
    if isfile(cacheFile)
        fprintf("Loading cached training data...\n");
        load(cacheFile, "inputImages", "wholeTumorMasks", ...
            "trainIdx", "validationIdx", "testIdx", "segFiles");
    else
        [inputImages, wholeTumorMasks, trainIdx, validationIdx, testIdx, segFiles] = patientTrainingDataRun(segmentFolder, outputFolder);
    
        save(cacheFile, "inputImages", "wholeTumorMasks", ...
            "trainIdx", "validationIdx", "testIdx", "segFiles", "-v7.3")
    end
end

function [nnLayers, nnModel] = genNNModel()
    nnLayers = [
        featureInputLayer(4, Normalization = "zscore")
        fullyConnectedLayer(16)
        reluLayer
        fullyConnectedLayer(8)
        reluLayer
        fullyConnectedLayer(2)
        softmaxLayer
        ];
    
    nnModel = dlnetwork(nnLayers);
end

function [X, Y] = samplePixelData(inputImages, wholeTumorMasks, sliceIdx, samplesPerClass)
    imgs = inputImages(:, :, :, sliceIdx);
    
    pixelLabels = categorical(wholeTumorMasks(:, :, sliceIdx), ...
        [false true], ["Background", "Tumor"]);
    pixelLabels = pixelLabels(:);
    pixelFeatures = reshape(single(imgs), [], 4);
    classIndices = [find(pixelLabels == "Background"); find(pixelLabels == "Tumor")];
    selectedIndices = arrayfun(@(c) randsample(find(pixelLabels == categorical(c)), ...
        min(samplesPerClass, nnz(pixelLabels == categorical(c)))), 1:2, "UniformOutput", false);

    % One row per pixel; four cols in t1n, t1c, t2w, t2f order
    pixelFeatures = reshape(permute(single(imgs), [1 2 4 3]), [], 4) / 255;

    pixelLabels = reshape(wholeTumorMasks(:, :, sliceIdx), [], 1);
    brainPixels = reshape(any(imgs > 0, 3), [], 1);

     tumorPixels = find(brainPixels & pixelLabels);
        bgPixels = find(brainPixels & ~pixelLabels);
     % Background Pixels

    n = min([samplesPerClass, numel(tumorPixels), numel(bgPixels)]);

    rng(1);
    tumorPixels      = tumorPixels(randperm(numel(tumorPixels), n));
    % Background Pixels 
    bgPixels = bgPixels(randperm(numel(bgPixels), n));

    selectedPixels = [tumorPixels; bgPixels];
    X = pixelFeatures(selectedPixels, :);

    labels = [ones(n, 1); zeros(n, 1)];
    Y = categorical(labels, [0 1], ["Background", "Tumor"]);
end

function [nnModel, trainInfo] = NeuralNetwork(inputImages, wholeTumorMasks, trainIdx, validationIdx, modelCacheFile)
    if isfile(modelCacheFile)
        cacheModel = load(modelCacheFile, "nnModel", "trainInfo");
        nnModel = cacheModel.nnModel;
        trainInfo = cacheModel.trainInfo;
        fprintf("Loading cached Neural Network....\n");
        return;
    else
        [nnModel, trainInfo] = trainNeuralNetwork(inputImages, wholeTumorMasks, trainIdx, validationIdx);
        save(modelCacheFile, "nnModel", "trainInfo", "-v7.3");
        fprintf("Neural Network training Complete and Cached.\n");
    end
end

function [nnModel, trainInfo] = trainNeuralNetwork(inputImages, wholeTumorMasks, trainIdx, validationIdx)
    [XTrain, YTrain] ... 
        = samplePixelData(inputImages, wholeTumorMasks, trainIdx, 10000);
    [XValidation, YValidation] ...
        = samplePixelData(inputImages, wholeTumorMasks, validationIdx, 5000);

    [nnLayers, nnModel] = genNNModel();

    options = trainingOptions( ...
                    "adam", ...
               MaxEpochs = 15, ...
           MiniBatchSize = 512, ...
        InitialLearnRate = 1e-3, ...
          ValidationData = {XValidation, YValidation}, ...
                   Plots = "training-progress", ...
     ValidationPatience = 3, ...
          OutputNetwork = "best-validation", ...
       L2Regularization = 2e-4 ...
    );

    [nnModel, trainInfo] = trainnet(XTrain, YTrain, nnModel, ...
        "crossentropy", options);
    fprintf("Neural network training complete.\n");
end

function printSlices(ModelName, inputImages, wholeTumorMasks, validationPreds, sliceIdx, displayIdx)
    close all;
    figure;
    tiledlayout(1, 3);
    
    nexttile;
    imshow(inputImages(:, :, 2, sliceIdx), []);
    title("T1C");

    nexttile;
    imshow(wholeTumorMasks(:, :, sliceIdx));
    title("Ground Truth");

    nexttile;
    imshow(validationPreds(:, :, displayIdx), []);
    title(ModelName);
    % title("NN Prediction");
end

function verifySlice(ModelName, inputImages, wholeTumorMasks, validationPreds, validationIdx, segFiles)
    validationSlices = find(validationIdx);
    targetName = "BraTS-MET-00008-000_BraTS-MET-00008-000-seg_slice_053.png";
    % hasTumor = squeeze(any(any( ...
        % wholeTumorMasks(:, :, validationSlices), 1), 2));  Too narrow of a check
    % displayIdx = find(hasTumor, 1);
    % sliceIdx = validationSlices(displayIdx);
    sliceIdx = find(contains({segFiles.name}, targetName), 1 );
    if isempty(sliceIdx)
        error("Could not find the GroundTruth file for the Slice 053");
    end

    displayIdx = find(validationSlices == sliceIdx, 1);
    if isempty(displayIdx)
        error("Slice 053 is not in the Validation Set");
    end
    printSlices(ModelName, inputImages, wholeTumorMasks, validationPreds, sliceIdx, displayIdx);
end

function validScores = validateSlices(slices, nnModel, inputImages, wholeTumorMasks)
    predictFcn = @(features) minibatchpredict( ...
        nnModel, features, MiniBatchSize=2048);
    
    validScores = validateCLASS( ...
        slices, nnModel, inputImages, wholeTumorMasks, predictFcn, 2);
end

function validScores = validateBoosting(slices, boostModel, inputImages, wholeTumorMasks)
    classNames = string(boostModel.ClassNames);
    tumorColumn = find(classNames == "Tumor", 1);

    % Intentionally Embedded
    function scores = predictBoostScores(boostModel, features)
        [~, scores] = predict(boostModel, features);
    end
    
    predictFcn = @(features) predictBoostScores(boostModel, features);
    
    validScores = validateCLASS( ...
        slices, boostModel, inputImages, wholeTumorMasks, ...
        predictFcn, tumorColumn);
end

function validScores = validateCLASS( ...
    slices, model, inputImages, wholeTumorMasks, predictFcn, tumorColumn)
    
    ht = size(wholeTumorMasks, 1);
    width = size(wholeTumorMasks, 2);
    validScores = zeros(ht, width, numel(slices), "single");
    
    for i = 1:numel(slices)
        sliceIdx = slices(i);
        img = inputImages(:, :, :, sliceIdx);
        brainPixels = find(any(img > 0, 3));
    
        if ~isempty(brainPixels)
            pixelFeatures = reshape(single(img), [], 4) / 255;
            scores = predictFcn(pixelFeatures(brainPixels, :));
    
            pixelScores = zeros(ht * width, 1, "single");
            pixelScores(brainPixels) = scores(:, tumorColumn);
            validScores(:, :, i) = reshape(pixelScores, ht, width);
        end
    
        if mod(i, 25) == 0 || i == numel(slices)
            fprintf("Evaluated %d of %d slices\n", i, numel(slices));
        end
    end
end

function [validationPreds, validationDice, bestThreshold] = validateByThreshold(validationScores, wholeTumorMasks, validationSlices)
    
    validationGroundTruth = wholeTumorMasks(:, :, validationSlices);
    thresholds = 0.10:0.05:0.90;

     diceByThreshold = zeros(size(thresholds));
    
    for i = 1:numel(thresholds)
        pred = validationScores >= thresholds(i);
        denom = nnz(pred) + nnz(validationGroundTruth);

        if denom > 0
            diceByThreshold(i) = ...
                2 * nnz(pred & validationGroundTruth) / denom;
        end
    end
    [validationDice, bestIdx] = max(diceByThreshold);
    bestThreshold = thresholds(bestIdx);

    fprintf("Best validation Dice: %.4f at threshold %.2f\n", ...
        validationDice, bestThreshold);

    validationPreds = validationScores >= bestThreshold;

end

function [validationPreds, validationDice] = evalNeuralNetwork(nnModel, inputImages, wholeTumorMasks, validationIdx, segFiles)
    validationSlices = find(validationIdx);
   
    % Run inference once and keep the probability maps
    validationScores = validateSlices(validationSlices, nnModel, inputImages, wholeTumorMasks);
    
    % Choose a threshold and get binary masks for scoring and display
    [validationPreds , validationDice] = validateByThreshold(validationScores, wholeTumorMasks, validationSlices);

    verifySlice("NN Prediction",inputImages, wholeTumorMasks, validationPreds, validationIdx, segFiles);
end

function runNeuralNetwork(inputImages, wholeTumorMasks, trainIdx, testIdx, validationIdx, modelCacheFile, segFiles, outputFolder)

    [nnModel, trainInfo] = NeuralNetwork(inputImages, wholeTumorMasks, trainIdx, validationIdx, modelCacheFile);
    
    [validationPreds, validationDice] = evalNeuralNetwork(nnModel, inputImages, wholeTumorMasks, validationIdx, segFiles);
    % disp(validationPreds) Prints 240 x 240 x 155 
    disp(validationDice)
    
    testSlices = find(testIdx);
    testScores = validateSlices(testSlices, nnModel, inputImages, wholeTumorMasks);
    testGroundTruth = wholeTumorMasks(:, :, testSlices);
    testPreds = testScores >= 0.55;
    
    testDenom = nnz(testPreds) + nnz(testGroundTruth);
    testDice = 2 * nnz(testPreds & testGroundTruth) / testDenom;
    
    fprintf("Test whole-tumor Dice at Threshold .55. %.4f\n", testDice);
    
    
    files = dir(fullfile(outputFolder, '*', '*00008*slice_053.png'));
    names = string({files.name});
    
    t1cIdx = find(contains(names, "-t1c_"), 1);
    segIdx = find(contains(names, "-seg_"), 1);
    
    t1c = imread(fullfile(files(t1cIdx).folder, files(t1cIdx).name));
    gt  = imread(fullfile(files(segIdx).folder, files(segIdx).name));
    
    fprintf("Ground-truth tumor pixels on slice 053: %d\n", nnz(gt > 0));
    
    % close all;
    % figure;
    % tiledlayout(1, 2);
    % nexttile; imshow(t1c, []); title("T1C slice 053");
    % nexttile; imshow(gt > 0); title("Ground truth slice 053");
end

function boostModel = BoostModel(inputImages, wholeTumorMasks, trainIdx)
    [XTrain, YTrain] = samplePixelData( ...
        inputImages, wholeTumorMasks, trainIdx, 10000);

    tree = templateTree(MaxNumSplits = 10);

    boostModel = fitcensemble(XTrain, YTrain,...
        Method = "LogitBoost", ...
        Learners = tree, ...
        NumLearningCycles = 100, ... 
        LearnRate = 0.1, ...
        ScoreTransform = "doublelogit" ...
        );
end

function U_Network(intputImages, trainIdx)
    % Separate training and Validation Images
    XTrain = single(inputImages(:, :, :, trainIdx)) / 255;
    XValidation = single(inputImages(:, :, :, validationIdx)) / 255;
    
    % Convert binary Masks to Pixel labels: Bg or Fg: Background/Tumor
    YTrain = categorical(uint8(wholeTumor(:, :, trainIdx)), ...
        [0 1], ["Background", "Tumor"]);
    YValidation = categorical(uint8(wholeTumorMasks(:, :, validationIdx)), ...
        [0 1], ["Background", "Tumor"]);
    
    % Add the label-channel dims expected for image segmentation 
    YTrain      = reshape(YTrain, size(YTrain, 1), size(YTrain, 2), 1, []);
    YValidation = reshape(YValidation, size(YValidation, 1), size(YValidation, 2), 1, []);
    
    % 
    % Create a compact, two-class U-Net for 240x240 image with 4 MRI channels
    baselineNet = unet([240 240 4], 2, ...
        EncoderDepth = 3, NumFirstEncoderFilters=16);
    
    size(XTrain)
    size(YTrain)
    size(XValidation)
    size(YValidation)
end

function roll = dice(pred, gt)
    roll = 2*nnz(pred & gt) / (nnz(pred) + nnz(gt));
end

function score = jaccard(pred, gt)
    score = nnz(pred & gt) / nnz(pred | gt);
end

function path = deterPath(name, Joseph_sharedPath, Aaron_sharedPath, Neeka_sharedPath)
    if(name == "Joseph")
        path = Joseph_sharedPath;
    elseif(name == "Aaron")
        path = Aaron_sharedPath;
    elseif(name == "Neeka")
        path = Neeka_sharedPath;
    else
        path = "INVALID";
    end
end


% main % main % main % main % main % main % main % main % main % main % main % main % main  
                                % Lol JAN

% Loading in data, for now replace the below with your own local files
Joseph_sharedPath = "/Users/josephrexrode/UTDallas/2026-2027/MATH6346/FinalProject";
Aaron_sharedPath = "C:/Users/aaron/Downloads/School/UTD/Medical Image Analysis/Final Project/";
Neeka_sharedPath = "/Users/neeka/Documents/medicalimage analysis/finalproject";

name = "Aaron";
sharedPath = deterPath(name, Joseph_sharedPath, Aaron_sharedPath, Neeka_sharedPath);

% Extract file paths
parentFolder  = fullfile(sharedPath,   "BraTS2026subset");
outputFolder  = fullfile(sharedPath,   "2DSlices");
rawFolder     = fullfile(outputFolder, "raw");
segmentFolder = fullfile(outputFolder, "groundTruth");

PNGFiles = dir(fullfile(outputFolder, '*', '*00008*slice_053.png'));

% segFiles = dir(fullfile(segmentFolder, '*seg_slice_*.png'));

% Run slicing Algos: (Multi)Otsu
% niiFiles = loadFiles(parentFolder, outputFolder, rawFolder, segmentFolder);
% convert2DFiles(niiFiles, rawFolder, segmentFolder);
% runSliceTest(outputFolder);


% Machine Learning Training Set
[inputImages, wholeTumorMasks, trainIdx, validationIdx, testIdx, segFiles] = ...
    patientTrainingData(segmentFolder, outputFolder);

trainingMasks = wholeTumorMasks(:, :, trainIdx);
nnz(trainingMasks) / numel(trainingMasks);

% [stack, groundTruth] = loadSliceStack(outputFolder, "00002", 1);
% [stack, groundTruth] = loadSliceStack(outputFolder, "00008", 53);
% unique(groundTruth)
% nnz(groundTruth > 0)

modelCacheFile = fullfile(outputFolder, "pixelNN.mat");
runNeuralNetwork(inputImages, wholeTumorMasks, trainIdx, testIdx, validationIdx, modelCacheFile, segFiles, outputFolder);

% boosting
boostModel = BoostModel(inputImages, wholeTumorMasks, trainIdx);

validationSlices = find(validationIdx);
boostValidationScores = validateBoosting( ...
    validationSlices, boostModel, inputImages, wholeTumorMasks);

[validationPreds, validationDice, bestThreshold] = validateByThreshold( ...
    boostValidationScores, wholeTumorMasks, validationSlices...
);

fprintf("Boosting validation Dice: %.4f\n", boostValidationDice);
verifySlice("Boosting Prediction", inputImages, wholeTumorMasks, boostValidationPreds, validationIdx, segFiles);


