%% tumor_segmentation_analysis.m
% Whole-tumor segmentation of BraTS-MET brain MRI using 2-D slices of four
% modalities (t1n, t1c, t2w, t2f).
%
% Pipeline
%   1. Baseline methods (Otsu, multi-Otsu, k-means, fuzzy c-means) on ONE
%      slice (patient 00008, slice 53). A demo only: not comparable to
%      ML numbers below.
%   2. Build 2-D slice dataset directly from the NIfTI volumes
%      (using buildSliceDataset.m: per-patient normalization).
%   3. Optional boosting baseline (issue: predicts tumor everywhere).
%   4. Pixel-wise NN (4 intensities per pixel) and U-Net
%      (using build_U_Network_Dice.m: Dice + cross-entropy loss). 
%      Both methods are cached.
%   5. compareModels.m: validation/test table. Thresholds chosen on
%      the validation patient only, then applied to the test patient.
%
% Data split (pt): train 00002/00004/00006, validation 00008, test 00009.
% Delete cached .mat files in the 2DSlices folder whenever the data or
% the model code changes (trainingData.mat, pixelNN.mat, unet_dice.mat).

%% Configs

RUN_CLASSICAL_BASELINES = true;   % re-converts every NIfTI to PNG on each run (slow)
RUN_BOOSTING            = false;  % known-broken baseline; set to false to skip

name = "Joseph";                  % "Joseph" | "Aaron" | "Neeka"

% Replace with local paths
Joseph_sharedPath = "/Users/josephrexrode/UTDallas/2026-2027/MATH6346/FinalProject";
Aaron_sharedPath  = "C:/Users/aaron/Downloads/School/UTD/Medical Image Analysis/Final Project/";
Neeka_sharedPath  = "/Users/neeka/Documents/medicalimage analysis/finalproject";
sharedPath = deterPath(name, Joseph_sharedPath, Aaron_sharedPath, Neeka_sharedPath);

parentFolder  = fullfile(sharedPath,   "BraTS2026subset");   % NIfTI volumes
outputFolder  = fullfile(sharedPath,   "2DSlices");          % PNGs + caches
rawFolder     = fullfile(outputFolder, "raw");
segmentFolder = fullfile(outputFolder, "groundTruth");

trainPatients     = ["00002", "00004", "00006"];    % patients used to train
validationPatient = "00008";                        % to validate
testPatient       = "00009";                        % to test

close all;   % close figures from previous runs
clc;         % clear the console

if ~exist(outputFolder, "dir")
    mkdir(outputFolder);
end

%% 1. Classical methods - baseline (on single slice)

if RUN_CLASSICAL_BASELINES
    niiFiles = loadFiles(parentFolder, outputFolder, rawFolder, segmentFolder);
    convert2DFiles(niiFiles, rawFolder, segmentFolder);
    runSliceTest(outputFolder);
end

%% 2. Dataset (2-D slices, normalized once per patient volume)

[inputImages, groundTruthLabels, wholeTumorMasks, ...
    trainIdx, validationIdx, testIdx, segFiles] = ...
        patientTrainingData(parentFolder, outputFolder, ...
        trainPatients, validationPatient, testPatient);

printClassCounts(groundTruthLabels);

%% 3. Boosting baseline (optional)

if RUN_BOOSTING
    runBoostingBaseline(inputImages, wholeTumorMasks, ...
        trainIdx, validationIdx, testIdx);
end

%% 4. Pixel NN and U-Net (trains once, then loads from cache file)

pixelNNCacheFile = fullfile(outputFolder, "pixelNN.mat");
uNetCacheFile    = fullfile(outputFolder, "unet_dice.mat");

[~, ~] = NeuralNetwork(inputImages, wholeTumorMasks, ...
    trainIdx, validationIdx, pixelNNCacheFile);

[uNet, uNetTrainInfo] = U_Network( ...
    inputImages, segFiles, wholeTumorMasks, trainIdx, ...
    validationIdx, uNetCacheFile);

%% 5. Final comparison table (validation-chosen thresholds, test results)

results = compareModels(inputImages, wholeTumorMasks, validationIdx, testIdx, ...
    pixelNNCacheFile, uNetCacheFile);


%% ========================================================================
%  LOCAL FUNCTIONS
%  A. Setup and data          B. Classical baselines (single slice)
%  C. Pixel-wise models       D. U-Net
%  E. Display                 F. Metrics
%  ========================================================================

%% ---- A. Setup and data -------------------------------------------------

function path = deterPath(name, Joseph_sharedPath, Aaron_sharedPath, Neeka_sharedPath)
    % Picks the project folder for whoever is running the script
    if name == "Joseph"
        path = Joseph_sharedPath;
    elseif name == "Aaron"
        path = Aaron_sharedPath;
    elseif name == "Neeka"
        path = Neeka_sharedPath;
    else
        error("Could not find the path to project files");
    end
end

function [inputImages, groundTruthLabels, wholeTumorMasks, ...
    trainIdx, validationIdx, testIdx, segFiles] = ...
    patientTrainingData(parentFolder, outputFolder, ...
    trainPatients, validationPatient, testPatient)
    % Loads the slice dataset from trainingData.mat if it is current,
    % otherwise rebuilds it with buildSliceDataset and saves it.

    cacheFile = fullfile(outputFolder, "trainingData.mat");
    dataCacheVersion = 3;
    patientConfig = [string(trainPatients(:)); "|"; ...
        string(validationPatient); "|"; string(testPatient)];

    requiredNames = ["dataCacheVersion", "patientConfig", "inputImages", ...
        "groundTruthLabels", "wholeTumorMasks", "trainIdx", ...
        "validationIdx", "testIdx", "segFiles"];

    useCache = false;
    if isfile(cacheFile)
        cacheInfo = whos("-file", cacheFile);
        if all(ismember(requiredNames, string({cacheInfo.name})))
            cached = load(cacheFile, requiredNames{:});
            useCache = cached.dataCacheVersion == dataCacheVersion && ...
                isequal(cached.patientConfig, patientConfig);
        end
    end

    if useCache
        fprintf("Loading cached training data...\n");
        inputImages       = cached.inputImages;
        groundTruthLabels = cached.groundTruthLabels;
        wholeTumorMasks   = cached.wholeTumorMasks;
        trainIdx          = cached.trainIdx;
        validationIdx     = cached.validationIdx;
        testIdx           = cached.testIdx;
        segFiles          = cached.segFiles;
    else
        fprintf("Building slice dataset from the NIfTI volumes...\n");
        [inputImages, groundTruthLabels, wholeTumorMasks, ...
            trainIdx, validationIdx, testIdx, segFiles] = ...
            buildSliceDataset(parentFolder, trainPatients, ...
            validationPatient, testPatient);

        save(cacheFile, "dataCacheVersion", "patientConfig", ...
            "inputImages", "groundTruthLabels", "wholeTumorMasks", ...
            "trainIdx", "validationIdx", "testIdx", "segFiles", "-v7.3");
    end
end

function printClassCounts(groundTruthLabels)
    % BraTS-MET label values: 0 background, 1 non-enhancing core,
    % 2 FLAIR hyperintensity (edema), 3 enhancing tumor
    classNames = ["Background"; "NonEnhancingTumorCore"; ...
        "FLAIRHyperintensity"; "EnhancingTumor"];
    for k = 1:numel(classNames)
        fprintf("%s (ID %d): %d pixels\n", classNames(k), k - 1, ...
            nnz(groundTruthLabels == k - 1));
    end
end

%% ---- B. Classical baselines (single slice, uses PNG slices) ------------
% These work on PNGs that were rescaled per slice with mat2gray, so they
% keep old normalization. The segmentation PNGs therefore hold
% 0/85/170/255 (only if the slice contains label 3), which is why
% runSliceTest compares against 255.

function niiFiles = loadFiles(parentFolder, outputFolder, rawFolder, segmentFolder)
    % Creates the output folders if needed and lists all 3-D NIfTI files
    if ~exist(outputFolder, "dir")
        mkdir(outputFolder);
        mkdir(segmentFolder);
        mkdir(rawFolder);
    end
    niiFiles = dir(fullfile(parentFolder, '**', '*.nii.gz'));
end

function convert2DFiles(niiFiles, rawFolder, segmentFolder)
    % Converts each 3-D volume into one PNG per slice (many small files)
    for i = 1:length(niiFiles)
        filePath = fullfile(niiFiles(i).folder, niiFiles(i).name);
        vol = niftiread(filePath);

        [~, baseName, ~] = fileparts(niiFiles(i).name);
        [~, baseName, ~] = fileparts(baseName);          % strip ".nii" and ".gz"
        [~, subFolderName] = fileparts(niiFiles(i).folder);

        isSegmentation = endsWith(baseName, "seg");

        for j = 1:size(vol, 3)
            sliceNorm = mat2gray(vol(:, :, j));
            outName = sprintf('%s_%s_slice_%03d.png', subFolderName, baseName, j);

            if isSegmentation
                imwrite(sliceNorm, fullfile(segmentFolder, outName));
            else
                imwrite(sliceNorm, fullfile(rawFolder, outName));
            end
        end
    end
end

function runSliceTest(outputFolder)
    % Runs every classical method on slice 53 of patient 00008 (all four
    % modalities), prints scores, and shows comparison figures.
    cacheFolder = fullfile(outputFolder, "model_cache");
    if ~exist(cacheFolder, "dir")
        mkdir(cacheFolder);
    end

    testMatchingFiles = dir(fullfile(outputFolder, '*', '*00008*slice_053.png'));

    % The first match is the ground-truth segmentation (dir lists the
    % "groundTruth" folder before "raw"). The scoring loops therefore start at 2.
    groundTruth = imread(fullfile( ...
        testMatchingFiles(1).folder, testMatchingFiles(1).name));

    wholeTumor = groundTruth > 0;
    enhancing  = groundTruth == 255;

    otsuMasks = cachedScoreSliceImages( ...
        "Otsu", "otsu_slice053_v1", @OtsuMethod, ...
        testMatchingFiles, enhancing, wholeTumor, cacheFolder);

    multiOtsuMasks = cachedScoreSliceImages( ...
        "Multi-Otsu", "multi_otsu_slice053_v1", @MultiOtsuMethod, ...
        testMatchingFiles, enhancing, wholeTumor, cacheFolder);

    kMeansMasks = runKmeans(testMatchingFiles, enhancing, wholeTumor, cacheFolder);

    fuzzyMasks = cachedScoreSliceImages( ...
        "Fuzzy C-Means", "fuzzy_cmeans_k4_slice053_v1", ...
        @(img) FuzzyCMeansMethod(img, 4), ...
        testMatchingFiles, enhancing, wholeTumor, cacheFolder);

    displayK = 4;
    showSliceComparison(testMatchingFiles, wholeTumor, otsuMasks, ...
        multiOtsuMasks, kMeansMasks{displayK}, displayK, fuzzyMasks);
end

function kMeansMasks = runKmeans(testMatchingFiles, enhancing, wholeTumor, cacheFolder)
    kMeansMasks = cell(1, 5);
    for k = 2:5
        methodName = sprintf("KMeans k=%d", k);
        cacheKey   = sprintf("kmeans_k%d_slice053_v1", k);

        kMeansMasks{k} = cachedScoreSliceImages( ...
            methodName, cacheKey, @(img) KMeansMethod(img, k), ...
            testMatchingFiles, enhancing, wholeTumor, cacheFolder);
    end
end

function predictions = cachedScoreSliceImages( ...
    methodName, cacheKey, segmentMethod, testMatchingFiles, ...
    enhancing, wholeTumor, cacheFolder)
    % Scores a segmentation method on the test slices, using a cache file.
    % The cache counts as stale when the source PNGs changed (path, size or
    % date).

    cacheFile   = fullfile(cacheFolder, cacheKey + ".mat");
    sourcePaths = string(fullfile( ...
        {testMatchingFiles.folder}, {testMatchingFiles.name}));
    sourceBytes = [testMatchingFiles.bytes];
    sourceDates = [testMatchingFiles.datenum];

    if isfile(cacheFile)
        fprintf("Opening cache for %s:\n  %s\n", methodName, cacheFile);
        cache = load(cacheFile, "predictions", "cacheKey", ...
            "sourcePaths", "sourceBytes", "sourceDates");

        cacheIsCurrent = isfield(cache, "predictions") && ...
            isfield(cache, "cacheKey") && ...
            isfield(cache, "sourcePaths") && ...
            isfield(cache, "sourceBytes") && ...
            isfield(cache, "sourceDates") && ...
            cache.cacheKey == cacheKey && ...
            isequal(cache.sourcePaths, sourcePaths) && ...
            isequal(cache.sourceBytes, sourceBytes) && ...
            isequal(cache.sourceDates, sourceDates);

        if cacheIsCurrent
            predictions = cache.predictions;
            fprintf("Using cached %s results from:\n  %s\n", methodName, cacheFile);
            printCachedSliceScores(methodName, testMatchingFiles, ...
                predictions, enhancing, wholeTumor);
            return;
        end

        fprintf("Cache is stale; recomputing %s results.\n", methodName);
    else
        fprintf("No cache for %s; computing results for:\n  %s\n", ...
            methodName, cacheFile);
    end

    predictions = scoreSliceImages(methodName, segmentMethod, ...
        testMatchingFiles, enhancing, wholeTumor);

    save(cacheFile, "predictions", "cacheKey", ...
        "sourcePaths", "sourceBytes", "sourceDates", "-v7.3");
    fprintf("Saved %s cache to:\n  %s\n", methodName, cacheFile);
end

function predictions = scoreSliceImages(methodName, segmentMethod, ...
    testMatchingFiles, enhancing, wholeTumor)
    predictions = cell(size(testMatchingFiles));

    for i = 2:numel(testMatchingFiles)       % file 1 is ground truth
        file = testMatchingFiles(i);
        img  = imread(fullfile(file.folder, file.name));

        pred = segmentMethod(img);
        predictions{i} = pred;
        printScores(methodName, file.name, pred, enhancing, wholeTumor);
    end
    fprintf("\n");
end

function printCachedSliceScores(methodName, files, predictions, enhancing, wholeTumor)
    for i = 2:numel(files)
        if ~isempty(predictions{i})
            printScores(methodName, files(i).name, predictions{i}, ...
                enhancing, wholeTumor);
        end
    end
    fprintf("\n");
end

function printScores(methodName, fileName, pred, enhancing, wholeTumor)
    fprintf("%s | %s | Dice vs Enhancing: %.3f | Jaccard vs Enhancing: %.3f | Dice vs Whole: %.3f\n", ...
        methodName, fileName, dice(pred, enhancing), ...
        jaccard(pred, enhancing), dice(pred, wholeTumor));
    printMetrix(methodName + " vs Enhancing", pred, enhancing);
    printMetrix(methodName + " vs Whole Tumor", pred, wholeTumor);
end

% --- The classical methods. Each takes one image and returns a logical mask.
% They all assume the brightest cluster is the tumor, which only makes sense
% for T1c (enhancing tumor) and FLAIR (edema).

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

function pred = FuzzyCMeansMethod(img, k)
    brainOnly = img > 0;
    x = double(img(brainOnly));

    options = fcmOptions(NumClusters = k, Verbose = false);
    [centers, U] = fcm(x, options);

    [~, tumorCluster] = max(centers);
    [~, labels] = max(U, [], 1);

    pred = false(size(img));
    pred(brainOnly) = labels == tumorCluster;
end

% --- Figures for the classical methods

function showSliceComparison(testMatchingFiles, wholeTumor, otsuMasks, ...
    multiOtsuMasks, kMeansDisplayMasks, displayK, fuzzyMasks)
    % One figure per modality: image, reference, then each method's mask.
    % T1c: T1 with contrast (brightens enhancing tumor)
    % T1n: T1 without contrast (anatomy)
    % T2w: T2-weighted (water-rich areas: edema and CSF)
    % T2f: T2-FLAIR (edema)
    fileNames = string({testMatchingFiles.name});

    for mode = ["t1c", "t2w", "t2f", "t1n"]
        suffix = "-" + mode + "_slice_053.png";
        idx = find(contains(fileNames, suffix), 1);

        if isempty(idx)
            warning("No matching PNG found for %s.", mode);
            continue
        end

        file = testMatchingFiles(idx);
        img  = imread(fullfile(file.folder, file.name));

        figure("Name", file.name, "NumberTitle", "off");
        tiledlayout(1, 6, "TileSpacing", "Compact");
        showTile(img, upper(mode) + " MRI", true);
        showTile(wholeTumor, sprintf("Reference:\nWhole Tumor"), false);
        showTile(otsuMasks{idx}, "Otsu", false);
        showTile(multiOtsuMasks{idx}, "Multi-Otsu", false);
        showTile(kMeansDisplayMasks{idx}, sprintf("K-means\n(k=%d)", displayK), false);
        showTile(fuzzyMasks{idx}, "Fuzzy C-Means (k=4)", false);
    end
end

function showTile(img, titleText, autoScale)
    nexttile;
    if autoScale
        imshow(img, []);
    else
        imshow(img);
    end
    title(titleText);
end

%% ---- C. Pixel-wise models (neural network, boosting) -------------------
% Each pixel is described only by its four intensities (t1n, t1c, t2w, t2f),
% so these models have no spatial context.

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
    % Draws a class-balanced random sample of brain pixels:
    % X is (2n x 4) intensities, Y is the matching Background/Tumor label.
    imgs = inputImages(:, :, :, sliceIdx);

    % One row per pixel; four columns in t1n, t1c, t2w, t2f order
    pixelFeatures = reshape(permute(single(imgs), [1 2 4 3]), [], 4);
    pixelLabels   = reshape(wholeTumorMasks(:, :, sliceIdx), [], 1);
    brainPixels   = reshape(any(imgs > 0, 3), [], 1);

    tumorPixels = find(brainPixels & pixelLabels);
    bgPixels    = find(brainPixels & ~pixelLabels);

    n = min([samplesPerClass, numel(tumorPixels), numel(bgPixels)]);

    rng(1);
    tumorPixels = tumorPixels(randperm(numel(tumorPixels), n));
    bgPixels    = bgPixels(randperm(numel(bgPixels), n));

    selectedPixels = [tumorPixels; bgPixels];
    X = pixelFeatures(selectedPixels, :);

    labels = [ones(n, 1); zeros(n, 1)];
    Y = categorical(labels, [0 1], ["Background", "Tumor"]);
end

function [nnModel, trainInfo] = NeuralNetwork(inputImages, wholeTumorMasks, ...
    trainIdx, validationIdx, modelCacheFile)
    % Loads the cached pixel NN, or trains it and caches it
    if isfile(modelCacheFile)
        cacheModel = load(modelCacheFile, "nnModel", "trainInfo");
        nnModel   = cacheModel.nnModel;
        trainInfo = cacheModel.trainInfo;
        fprintf("Loading cached Neural Network....\n");
    else
        [nnModel, trainInfo] = trainNeuralNetwork( ...
            inputImages, wholeTumorMasks, trainIdx, validationIdx);
        save(modelCacheFile, "nnModel", "trainInfo", "-v7.3");
        fprintf("Neural Network training Complete and Cached.\n");
    end
end

function [nnModel, trainInfo] = trainNeuralNetwork( ...
    inputImages, wholeTumorMasks, trainIdx, validationIdx)
    rng(1);
    [XTrain, YTrain] = samplePixelData(inputImages, wholeTumorMasks, trainIdx, 10000);
    [XValidation, YValidation] = samplePixelData(inputImages, wholeTumorMasks, validationIdx, 5000);

    [~, nnModel] = genNNModel();

    options = trainingOptions( ...
                    "adam", ...
               MaxEpochs = 15, ...
           MiniBatchSize = 512, ...
        InitialLearnRate = 1e-3, ...
          ValidationData = {XValidation, YValidation}, ...
                   Plots = "training-progress", ...
     ValidationPatience = 3, ...
          OutputNetwork = "best-validation", ...
       L2Regularization = 2e-4);

    [nnModel, trainInfo] = trainnet(XTrain, YTrain, nnModel, ...
        "crossentropy", options);
    fprintf("Neural network training complete.\n");
end

function boostModel = BoostModel(inputImages, wholeTumorMasks, trainIdx)
    [XTrain, YTrain] = samplePixelData( ...
        inputImages, wholeTumorMasks, trainIdx, 10000);

    tree = templateTree(MaxNumSplits = 10);

    boostModel = fitcensemble(XTrain, YTrain, ...
        Method = "LogitBoost", ...
        Learners = tree, ...
        NumLearningCycles = 100, ...
        LearnRate = 0.1, ...
        ScoreTransform = "doublelogit");
end

function runBoostingBaseline(inputImages, wholeTumorMasks, ...
    trainIdx, validationIdx, testIdx)
    % KNOWN ISSUE: this model gives identical results whatever the input
    % (8.3% validation Dice, 14.7% test Dice, recall 100%, precision 4-8%),
    % which means it predicts "tumor" for every brain pixel. Not fixed yet.
    validationSlices = find(validationIdx);
    testSlices       = find(testIdx);

    boostModel = BoostModel(inputImages, wholeTumorMasks, trainIdx);

    validationScores = validateBoosting( ...
        validationSlices, boostModel, inputImages, wholeTumorMasks);
    [validationPreds, ~, threshold] = validateByThreshold( ...
        validationScores, wholeTumorMasks, validationSlices, "Boosting");
    printMetrix("Boosting validation", validationPreds, ...
        wholeTumorMasks(:, :, validationSlices));

    testScores = validateBoosting( ...
        testSlices, boostModel, inputImages, wholeTumorMasks);
    printMetrix("Boosting test", testScores >= threshold, ...
        wholeTumorMasks(:, :, testSlices));
end

function validScores = validateBoosting(slices, boostModel, inputImages, wholeTumorMasks)
    classNames  = string(boostModel.ClassNames);
    tumorColumn = find(classNames == "Tumor", 1);

    function scores = predictBoostScores(model, features)
        [~, scores] = predict(model, features);
    end

    predictFcn = @(features) predictBoostScores(boostModel, features);
    validScores = validateCLASS(slices, inputImages, wholeTumorMasks, ...
        predictFcn, tumorColumn);
end

function validScores = validateCLASS(slices, inputImages, wholeTumorMasks, ...
    predictFcn, tumorColumn)
    % Runs a pixel-wise classifier over whole slices. Only brain pixels
    % (any modality > 0) are scored; everything else stays 0.
    ht    = size(wholeTumorMasks, 1);
    width = size(wholeTumorMasks, 2);
    validScores = zeros(ht, width, numel(slices), "single");

    for i = 1:numel(slices)
        img = inputImages(:, :, :, slices(i));
        brainPixels = find(any(img > 0, 3));

        if ~isempty(brainPixels)
            pixelFeatures = reshape(single(img), [], 4);
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

function [validationPreds, validationDice, bestThreshold] = ...
    validateByThreshold(validationScores, wholeTumorMasks, validationSlices, modelName)
    % Picks the probability threshold with the best Dice on the validation
    % patient, and returns the thresholded masks.
    if nargin < 4
        modelName = "Model";
    end

    validationGroundTruth = wholeTumorMasks(:, :, validationSlices);
    thresholds = 0.10:0.05:0.90;
    diceByThreshold = zeros(size(thresholds));

    for i = 1:numel(thresholds)
        pred  = validationScores >= thresholds(i);
        denom = nnz(pred) + nnz(validationGroundTruth);
        if denom > 0
            diceByThreshold(i) = 2 * nnz(pred & validationGroundTruth) / denom;
        end
    end

    [validationDice, bestIdx] = max(diceByThreshold);
    bestThreshold   = thresholds(bestIdx);
    validationPreds = validationScores >= bestThreshold;

    fprintf("%s validation Dice: %.2f%% at threshold %.2f\n", ...
        modelName, validationDice * 100, bestThreshold);
end

%% ---- D. U-Net ----------------------------------------------------------
% Training itself lives in build_U_Network_Dice.m.

function [uNet, trainInfo, validationPreds, validationDice, bestThreshold] = U_Network( ...
    inputImages, segFiles, wholeTumorMasks, trainIdx, validationIdx, modelCacheFile)
    % Loads or trains the U-Net, then checks it on the validation patient
    [uNet, trainInfo] = U_Network_switch(inputImages, wholeTumorMasks, ...
        trainIdx, validationIdx, modelCacheFile);

    [validationPreds, validationDice, bestThreshold] = checkUNetThreshold( ...
        uNet, inputImages, wholeTumorMasks, validationIdx, segFiles);
end

function [uNet, trainInfo] = U_Network_switch( ...
    inputImages, wholeTumorMasks, trainIdx, validationIdx, modelCacheFile)
    % Loads the cached U-Net if there is one, otherwise trains and caches it
    if isfile(modelCacheFile)
        cache = load(modelCacheFile, "uNet");
        uNet = cache.uNet;
        trainInfo = [];
        fprintf("Loading Cached U-net.... \n");
        return;
    end

    [uNet, trainInfo] = build_U_Network_Dice( ...
        inputImages, wholeTumorMasks, trainIdx, validationIdx, modelCacheFile);
end

function validScores = validateUNet(slices, uNet, inputImages)
    % Tumor probability map for each requested slice
    ht    = size(inputImages, 1);
    width = size(inputImages, 2);
    validScores = zeros(ht, width, numel(slices), "single");

    for i = 1:numel(slices)
        img = single(inputImages(:, :, :, slices(i)));
        img = reshape(img, ht, width, size(inputImages, 3), 1);

        scores = minibatchpredict(uNet, img, MiniBatchSize = 1);

        % Output channels are [Background, Tumor]; channel 2 = tumor score
        validScores(:, :, i) = single(scores(:, :, 2, 1));
    end
end

function [validationPreds, validationDice, bestThreshold] = checkUNetThreshold( ...
    uNet, inputImages, wholeTumorMasks, validationIdx, segFiles)
    % Chooses U-Net's threshold on the validation patient, prints
    % metrics, and plots slice 53 for visual check
    validationSlices = find(validationIdx);
    validScores = validateUNet(validationSlices, uNet, inputImages);

    thresholds = 0.10:0.05:0.90;
    diceByThreshold = zeros(size(thresholds));
    validationGroundTruth = wholeTumorMasks(:, :, validationSlices);

    for i = 1:numel(thresholds)
        pred  = validScores >= thresholds(i);
        denom = nnz(pred) + nnz(validationGroundTruth);
        if denom > 0
            diceByThreshold(i) = 2 * nnz(pred & validationGroundTruth) / denom;
        end
    end

    [validationDice, bestIdx] = max(diceByThreshold);
    bestThreshold   = thresholds(bestIdx);
    validationPreds = validScores >= bestThreshold;

    fprintf("U-Net validation Dice score: %.2f%% at threshold %.2f\n", ...
        validationDice * 100, bestThreshold);

    verifySlice("U-Net Prediction", inputImages, wholeTumorMasks, ...
        validationPreds, validationIdx, segFiles);

    printMetrix("U-Net validation", validationPreds, validationGroundTruth);
end

%% ---- E. Display --------------------------------------------------------

function verifySlice(modelName, inputImages, wholeTumorMasks, validationPreds, ...
    validationIdx, segFiles)
    % Shows validation patient 00008, slice 53: T1c, ground truth, prediction
    validationSlices = find(validationIdx);
    targetName = "BraTS-MET-00008-000_BraTS-MET-00008-000-seg_slice_053.png";

    sliceIdx = find(contains({segFiles.name}, targetName), 1);
    if isempty(sliceIdx)
        error("Could not find the GroundTruth file for the Slice 053");
    end

    displayIdx = find(validationSlices == sliceIdx, 1);
    if isempty(displayIdx)
        error("Slice 053 is not in the Validation Set");
    end

    printSlices(modelName, inputImages, wholeTumorMasks, validationPreds, ...
        sliceIdx, displayIdx);
end

function printSlices(modelName, inputImages, wholeTumorMasks, validationPreds, ...
    sliceIdx, displayIdx)
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
    title(modelName);
end

%% ---- F. Metrics --------------------------------------------------------

function roll = dice(pred, gt)
    denom = nnz(pred) + nnz(gt);
    if denom == 0
        roll = NaN;
    else
        roll = 2 * nnz(pred & gt) / denom;
    end
end

function score = jaccard(pred, gt)
    denom = nnz(pred | gt);
    if denom == 0
        score = NaN;
    else
        score = nnz(pred & gt) / denom;
    end
end

function pct = percentOrNaN(top, denom)
    if denom == 0
        pct = NaN;
    else
        pct = 100 * top / denom;
    end
end

function perc = makePercent(decimal, pts)
    if nargin < 2, pts = 4; end
    perc = string(sprintf("%.*f", pts, decimal * 100)) + "%";
end

function metrics = compareRegionPixels(pred, gt)
    assert(isequal(size(pred), size(gt)), ...
        "Prediction and ground truth must have the same size");

    pred = logical(pred);
    gt   = logical(gt);

    TP = nnz(pred & gt);
    FP = nnz(pred & ~gt);
    FN = nnz(~pred & gt);
    TN = nnz(~pred & ~gt);

    % Precision: of the pixels called tumor, how many are in the ground truth?
    metrics.PredictedInsideGT_pct  = percentOrNaN(TP, TP + FP);
    metrics.PredictedOutsideGT_pct = percentOrNaN(FP, TP + FP);

    % Recall: of all ground-truth tumor pixels, how many were found?
    metrics.GroundTruthRecovered_pct = percentOrNaN(TP, TP + FN);

    % Of all ground-truth background pixels, how many were correctly rejected?
    metrics.BackgroundCorrect_pct = percentOrNaN(TN, TN + FP);

    metrics.Dice    = dice(pred, gt);
    metrics.Jaccard = jaccard(pred, gt);
    metrics.TP = TP;
    metrics.FP = FP;
    metrics.FN = FN;
    metrics.TN = TN;
end

% "Metrix" is deliberate (a short "Metrics"); rename it if it confuses you.
function printMetrix(methodName, pred, gt)
    metrix = compareRegionPixels(pred, gt);

    fprintf("\n%s\n", methodName);
    fprintf("   Predicted Inside GT: %.2f%%\n", metrix.PredictedInsideGT_pct);
    fprintf("  Predicted Outside GT: %.2f%%\n", metrix.PredictedOutsideGT_pct);
    fprintf("Ground Truth Recovered: %.2f%%\n", metrix.GroundTruthRecovered_pct);
    fprintf("                  Dice: %s\n", makePercent(metrix.Dice, 2));
    fprintf("               Jaccard: %s\n", makePercent(metrix.Jaccard, 2));
end