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

function predictions = cachedScoreSliceImages( ...
    methodName, cacheKey, segmentMethod, testMatchingFiles, ...
    enhancing, wholeTumor, cacheFolder)

cacheFile = fullfile(cacheFolder, cacheKey + ".mat");
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
            fprintf("Using cached %s results from:\n  %s\n", ...
                methodName, cacheFile);
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

function printCachedSliceScores(methodName, files, predictions, enhancing, wholeTumor)
    for i = 2:numel(files)
        if ~isempty(predictions{i})
            printScores(methodName, files(i).name, predictions{i}, ...
                enhancing, wholeTumor);
        end
    end
    fprintf("\n");
end

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

function pred = FuzzyCMeansMethod(img, k, debugging)
    arguments
        img, k, debugging = false
    end

    brainOnly = img > 0;
    x = double(img(brainOnly));
    if debugging % the debugging printouts
        [centers, U] = fcm(x, k); % This line can be restored if needed
    else
        options = fcmOptions(NumClusters=k, Verbose=false);
        [centers, U] = fcm(x, options);
    end
    
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
    printMetrix(methodName + " vs Enhancing", pred, enhancing);
    printMetrix(methodName + " vs Whole Tumor", pred, wholeTumor);
end

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
    title(sprintf("Reference:\nWhole Tumor"));
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
    title(sprintf("Multimodal K-means\n(k=%d)", displayK));
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
    % close all;
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

function kMeansMasks = runKmeans(testMatchingFiles, enhancing, wholeTumor, cacheFolder)
    kMeansMasks = cell(1, 5);

    for k = 2:5
        methodName = sprintf("KMeans k=%d", k);
        cacheKey = sprintf("kmeans_k%d_slice053_v1", k);

        kMeansMasks{k} = cachedScoreSliceImages( ...
            methodName, cacheKey, @(img) KMeansMethod(img, k), ...
            testMatchingFiles, enhancing, wholeTumor, cacheFolder);
    end
end

function runSliceTest(outputFolder, method)
    arguments
       outputFolder,
       method ="All"
    end

    cacheFolder = fullfile(outputFolder, "model_cache");
    if ~exist(cacheFolder, "dir")
        mkdir(cacheFolder);
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
    
       otsuMasks = cachedScoreSliceImages( ...
        "Otsu", "otsu_slice053_v1", @OtsuMethod, ...
        testMatchingFiles, enhancing, wholeTumor, cacheFolder);
    
    multiOtsuMasks = cachedScoreSliceImages( ...
        "Multi-Otsu", "multi_otsu_slice053_v1", @MultiOtsuMethod, ...
        testMatchingFiles, enhancing, wholeTumor, cacheFolder);
    
    kMeansMasks = runKmeans(testMatchingFiles, enhancing, ...
        wholeTumor, cacheFolder);
    
    CfuzzyMasks = cachedScoreSliceImages( ...
        "Fuzzy C-Means", "fuzzy_cmeans_k4_slice053_v1", ...
        @(img) FuzzyCMeansMethod(img, 4), ...
        testMatchingFiles, enhancing, wholeTumor, cacheFolder);
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

function labelScheme = bratsMETLabelScheme()
    % Stable Class IDs, semantic names, and regions that include each class
    labelScheme = table(...
          uint8((0:3)'), ...
        ["Background"; "NonEnhancingTumorCore"; ...
         "FLAIRHyperintensity"; "EnhancingTumor"], ...
        {"Background"; ["TumorCore", "WholeTumor"]; ...
         ["Edema", "WholeTumor"]; ...
         ["EnhancingTumor", "TumorCore", "WholeTumor"]}, ...
        uint8([0; 1; 2; 3]), ...
        uint8([0; 85; 170; 255]), ...
        'VariableNames', ...
        {'ID', 'ClassName', 'Regions', 'SourceValue', 'LegacyPNGValue'})
end

function labelMap = decodeLabelMap(storedMask, labelScheme)
    % Accept original labels 0–3 or the old normalized PNG values.
    labelMap = zeros(size(storedMask), "uint8");
    recognized = false(size(storedMask));

    for k = 1:height(labelScheme)
        sourcePixels = storedMask == labelScheme.SourceValue(k);
        legacyPixels = storedMask == labelScheme.LegacyPNGValue(k);

        labelMap(sourcePixels | legacyPixels) = labelScheme.ID(k);
        recognized = recognized | sourcePixels | legacyPixels;
    end

    if ~all(recognized, "all")
        unexpected = unique(storedMask(~recognized));
        error("Unexpected segmentation pixel values: %s", ...
            mat2str(unexpected(:)'));
    end
end


function mask = makeRegionMask(labelMap, labelScheme, regionName)
    % Build a binary mask for any named region in the scheme.
    regionName = string(regionName);
    belongsToRegion = cellfun( ...
        @(regions) any(string(regions) == regionName), ...
        labelScheme.Regions);

    if ~any(belongsToRegion)
        error("Region '%s' is not defined in the label scheme.", regionName);
    end

    mask = ismember(labelMap, labelScheme.ID(belongsToRegion));
end
function [inputImage, groundTruthLabels, wholeTumor] = preprocessingSlice(PNGFiles, labelScheme)
    fileNames = string({PNGFiles.name});
    modes = ["t1n", "t1c", "t2w", "t2f"];

    firstIdx   = find(contains(fileNames, "-" + modes(1) + "_"), 1);
    % disp(firstIdx)
    firstImage = imread(fullfile(PNGFiles(firstIdx).folder, PNGFiles(firstIdx).name));
    inputImage = zeros([size(firstImage), 4], "like", firstImage);

    for c = 1:numel(modes)
        idx = find(contains(fileNames, "-" + modes(c) + "_"), 1);
        inputImage(:, :, c) = imread(fullfile(PNGFiles(idx).folder, PNGFiles(idx).name));
    end

    segIdx = find(contains(fileNames, "-seg_"), 1);
    storedMask = imread(fullfile(...
        PNGFiles(segIdx).folder, PNGFiles(segIdx).name));

    groundTruthLabels = decodeLabelMap(storedMask, labelScheme);
     wholeTumor  = makeRegionMask(groundTruthLabels, labelScheme, "WholeTumor");

    % groundTruth = imread(fullfile(PNGFiles(segIdx).folder, PNGFiles(segIdx).name));
    % wholeTumor  = groundTruth > 0;

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

function [inputImages, groundTruthLabels, wholeTumorMasks, ...
    trainIdx, validationIdx, testIdx, segFiles] = ... 
patientTrainingDataRun(segmentFolder, outputFolder)
    
    labelScheme = bratsMETLabelScheme();

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
    groundTruthLabels = zeros(240, 240, sF, "uint8");
    wholeTumorMasks = false(240, 240, sF);

    allPNGfiles = dir(fullfile(outputFolder, "*", "*slice_*.png"));
    allPNGnames = string({allPNGfiles.name});
    
    groundTruthLabels = zeros(240, 240, sF, "uint8");
    for i = 1:numel(segFiles)
        fileName  = string(segFiles(i).name);
        patientID = extractBetween(fileName, "BraTS-MET-", "-000_");
        sliceNum  = str2double(extractBetween(fileName, "_slice_", ".png"));

        sliceText = extractBetween(fileName, "_slice_", ".png");
        
        sliceFiles = allPNGfiles( ...
            contains(allPNGnames, "BraTS-MET-" + patientID + "-") & ...
            endsWith(allPNGnames, "_slice_" + sliceText + ".png"));

        [inputImage, groundTruthLabel, wholeTumor] = preprocessingSlice(sliceFiles, labelScheme);
        
        inputImages(:, :, :, i) = inputImage;
        groundTruthLabels(:, :, i) = groundTruthLabel;
        wholeTumorMasks(:, :, i) = wholeTumor;
    
        if mod(i, 25) == 0 || i == numel(segFiles)
            fprintf("Loaded %d of %d slices\n", i, numel(segFiles));
        end
    end    
end

function [inputImages, groundTruthLabels, wholeTumorMasks, ...
    trainIdx, validationIdx, testIdx, segFiles] = ...
    patientTrainingData(parentFolder, outputFolder)

    cacheFile = fullfile(outputFolder, "trainingData.mat");
    dataCacheVersion = 3;
    useCache = false;
    
    if isfile(cacheFile)
        cacheInfo = whos("-file", cacheFile);
        cacheNames = string({cacheInfo.name});
    
        requiredNames = ["dataCacheVersion", "inputImages", ...
            "groundTruthLabels", "wholeTumorMasks", "trainIdx", ...
            "validationIdx", "testIdx", "segFiles"];
    
        if all(ismember(requiredNames, cacheNames))
            cached = load(cacheFile, requiredNames{:});
            useCache = cached.dataCacheVersion == dataCacheVersion;
        end
    end
    
    if useCache
        fprintf("Loading cached training data...\n");
        inputImages = cached.inputImages;
        groundTruthLabels = cached.groundTruthLabels;
        wholeTumorMasks = cached.wholeTumorMasks;
        trainIdx = cached.trainIdx;
        validationIdx = cached.validationIdx;
        testIdx = cached.testIdx;
        segFiles = cached.segFiles;
    else
        fprintf("Rebuilding cache with categorical labels...\n");
        [inputImages, groundTruthLabels, wholeTumorMasks, ...
            trainIdx, validationIdx, testIdx, segFiles] = ...
            buildSliceDataset(parentFolder, ["00002","00004","00006"], "00008", "00009");
    
        save(cacheFile, "dataCacheVersion", "inputImages", ...
            "groundTruthLabels", "wholeTumorMasks", "trainIdx", ...
            "validationIdx", "testIdx", "segFiles", "-v7.3");
    end
end



    % else
    %     [inputImages, wholeTumorMasks, trainIdx, validationIdx, testIdx, segFiles] = patientTrainingDataRun(segmentFolder, outputFolder);
    % 
    %     save(cacheFile, "inputImages", "wholeTumorMasks", ...
    %         "trainIdx", "validationIdx", "testIdx", "segFiles", "-v7.3");

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
    pixelFeatures = reshape(permute(single(imgs), [1 2 4 3]), [], 4);

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
    % close all;
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
    
    if nargin < 4
        modelName = "Model";
    end

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

    fprintf("NN: 'Best Validation Score' Dice: %.4f at threshold %.2f\n", ...
        validationDice, bestThreshold);

    validationPreds = validationScores >= bestThreshold;

    fprintf("%s validation Dice: %.2f%% at threshold %.2f\n", ...
        modelName, validationDice * 100, bestThreshold);

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

    printMetrix("Neural Network", testPreds, testGroundTruth);

    testDenom = nnz(testPreds) + nnz(testGroundTruth);
    testDice = 2 * nnz(testPreds & testGroundTruth) / testDenom;
    
    fprintf("NN: 'Test Score' whole-tumor Dice at Threshold 55%%: %.4f\n", testDice);

    
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

function options = uNetModelOptions(XValidation, YValidation)
    options = trainingOptions("adam", ...
    MaxEpochs = 15,...
    MiniBatchSize = 2,...
    InitialLearnRate = 1e-3,...
    ValidationData = {XValidation, YValidation}, ...
    ValidationFrequency = 50, ...
    ValidationPatience = 3, ...
    OutputNetwork = "best-validation", ...
    Plots = "training-progress", ...
    Verbose = true...
    );
end

function [uNet, trainInfo] = build_U_Network( ...
    inputImages, wholeTumorMasks, trainIdx, validationIdx, modelCacheFile)
    
    XTrain = single(inputImages(:, :, :, trainIdx));
    XValidation = single(inputImages(:, :, :, validationIdx));
    
    % Convert binary Masks to Pixel labels: Bg or Fg: Background/Tumor
    YTrain = categorical(uint8(wholeTumorMasks(:, :, trainIdx)), ...
        [0 1], ["Background", "Tumor"]);
    YValidation = categorical(uint8(wholeTumorMasks(:, :, validationIdx)), ...
        [0 1], ["Background", "Tumor"]);
    
    % Add the label-channel dims expected for image segmentation 
    YTrain      = reshape(YTrain, size(YTrain, 1), size(YTrain, 2), 1, []);
    YValidation = reshape(YValidation, size(YValidation, 1), size(YValidation, 2), 1, []);
   
    % Create a compact, two-class U-Net for 240x240 image with 4 MRI channels
    baselineNet = unet([240 240 4], 2, ...
        EncoderDepth = 3, NumFirstEncoderFilters=16);
    
    options = uNetModelOptions(XValidation, YValidation);
    
    [uNet, trainInfo] = trainnet( ...
        XTrain, YTrain, baselineNet, "crossentropy", options);
    
    save(modelCacheFile, "uNet", "trainInfo", "-v7.3");
    fprintf("U-Net training complete and cached.\n");
    
    size(XTrain)
    size(YTrain)
    size(XValidation)
    size(YValidation)

end

function [uNet, trainInfo] = U_Network_switch( ...
inputImages , wholeTumorMasks, trainIdx, validationIdx, modelCacheFile)
    % Separate training and Validation Images
    
    if isfile(modelCacheFile)
         % cache = load(modelCacheFile, "uNet", "trainInfo");
         % uNet = cache.uNet;
         % trainInfo = cache.trainInfo;

         cache = load(modelCacheFile, "uNet");
         uNet = cache.uNet;
         trainInfo = [];

         fprintf("Loading Cached U-net.... \n");
         return;
    end

    [uNet, trainInfo] = build_U_Network( ...
        inputImages , wholeTumorMasks, trainIdx, validationIdx, modelCacheFile)
    
end

function validScores = validateUNet(slices, uNet, inputImages)
    ht = size(inputImages, 1);
    width = size(inputImages, 2);

    validScores = zeros(ht, width, numel(slices), "single");

    for i = 1:numel(slices)
        img = single(inputImages(:, :, :, slices(i)));
        img= reshape(img, ht, width, size(inputImages, 3), 1);

        scores = minibatchpredict(uNet, img, MiniBatchSize = 1);

        % Categories were created in Background, Tumor
        % The second output channel is the tumor score.
        validScores(:, :, i) = single(scores(:, :, 2, 1));
    end
end

function [validationPreds, validationDice, bestThreshold] = checkUNetThreshold(uNet, inputImages, wholeTumorMasks, validationIdx, segFiles)
    validationSlices = find(validationIdx);
    
    validScores = validateUNet(validationSlices, uNet, inputImages);
    
    thresholds = 0.10:0.05:0.90;
    diceByThreshold = zeros(size(thresholds));
    validationGroundTruth = wholeTumorMasks(:, :, validationSlices);
    
    for i = 1:numel(thresholds)
        pred = validScores >= thresholds(i);
        denom = nnz(pred) + nnz(validationGroundTruth);
    
        if denom > 0
            diceByThreshold(i) = ...
                2 * nnz(pred & validationGroundTruth) / denom;
        end
    end
    
    [validationDice, bestIdx] = max(diceByThreshold);
    bestThreshold = thresholds(bestIdx);
    validationPreds = validScores >= bestThreshold;
    
    fprintf("U-Net validation Dice score: %.2f%% at threshold %.2f\n", ...
        validationDice * 100, bestThreshold);
    
    verifySlice("U-Net Prediction", inputImages, wholeTumorMasks, ...
        validationPreds, validationIdx, segFiles);

    printMetrix("U-Net validation", validationPreds, validationGroundTruth);
end

function [uNet, trainInfo, validationPreds, validationDice, bestThreshold] = U_Network( ...
    inputImages, segFiles, wholeTumorMasks, trainIdx, validationIdx, modelCacheFile)

    [uNet, trainInfo] = U_Network_switch( inputImages , wholeTumorMasks, ...
        trainIdx, validationIdx, modelCacheFile);

    [validationPreds, validationDice, bestThreshold] = checkUNetThreshold(uNet, inputImages, wholeTumorMasks, validationIdx, segFiles);
    
end

function closedPreds = openAndClosedPreds(preds, radius)
    se = strel("disk", radius, 0);
    closedPreds = false(size(preds));

    for i = 1:size(preds, 3)
        mask = imopen(logical(preds(:, :, i)), se);
        closedPreds(:, :, i) = imclose(mask, se);
    end
end

function runClosedFusionAnalysis(uNet, unetValidationPreds, unetThreshold, ...
    nnValidationPreds, nnTestPreds, boostValidationPreds, boostTestPreds, ...
    inputImages, wholeTumorMasks, validationIdx, testIdx, segFiles)

    validationSlices = find(validationIdx);
    testSlices = find(testIdx);
    validationGT = wholeTumorMasks(:, :, validationSlices);
    testGT = wholeTumorMasks(:, :, testSlices);

    % Apply the U-Net threshold chosen on validation to test.
    unetTestScores = validateUNet(testSlices, uNet, inputImages);
    unetTestPreds = unetTestScores >= unetThreshold;

    % Close each method's masks separately.
    closeRadius = 1;
    nnValClosed = openAndClosedPreds(nnValidationPreds, closeRadius);
    boostValClosed = openAndClosedPreds(boostValidationPreds, closeRadius);
    unetValClosed = openAndClosedPreds(unetValidationPreds, closeRadius);

    nnTestClosed = openAndClosedPreds(nnTestPreds, closeRadius);
    boostTestClosed = openAndClosedPreds(boostTestPreds, closeRadius);
    unetTestClosed = openAndClosedPreds(unetTestPreds, closeRadius);

    % Choose the vote rule using validation only.
    votesVal = double(nnValClosed) + ...
        double(boostValClosed) + double(unetValClosed);
    voteRules = 1:3;
    diceByRule = zeros(size(voteRules));

    for i = 1:numel(voteRules)
        candidate = votesVal >= voteRules(i);
        denom = nnz(candidate) + nnz(validationGT);
        if denom > 0
            diceByRule(i) = 2 * nnz(candidate & validationGT) / denom;
        end
    end

    [bestDice, bestIdx] = max(diceByRule);
    bestRule = voteRules(bestIdx);
    fusionVal = votesVal >= bestRule;

    fprintf("Closed fusion validation Dice: %.2f%% using %d-of-3 votes\n", ...
        100 * bestDice, bestRule);
    printMetrix("Closed fusion validation", fusionVal, validationGT);

    % Apply the validation-selected rule to test.
    votesTest = double(nnTestClosed) + ...
        double(boostTestClosed) + double(unetTestClosed);
    fusionTest = votesTest >= bestRule;

    printMetrix("Closed fusion test", fusionTest, testGT);
    verifySlice("Closed Swiss-cheese fusion", inputImages, wholeTumorMasks, ...
        fusionVal, validationIdx, segFiles);
end

function runSegmentionModels(inputImages, wholeTumorMasks, trainIdx, ...
    validationIdx, testIdx, segFiles, outputFolder)

    validationSlices = find(validationIdx);
    testSlices = find(testIdx);
    
    % Pixel Neural Network: choose Threshold on Validation, then Freeze it
    pixelNNCacheFile = fullfile(outputFolder, "pixelNN.mat");
    [nnModel, ~] = NeuralNetwork(inputImages, wholeTumorMasks, trainIdx, validationIdx, pixelNNCacheFile);

    nnValidationScores = validateSlices( ...
        validationSlices, nnModel, inputImages, wholeTumorMasks);
    [nnValidationPreds, ~, nnThreshold] = validateByThreshold( ...
        nnValidationScores, wholeTumorMasks, validationSlices, ...
        "Pixel neural network");

nnTestScores = validateSlices( ...
        testSlices, nnModel, inputImages, wholeTumorMasks);
    nnTestPreds = nnTestScores >= nnThreshold;

    % Boosting: choose threshold on validation, then freeze it
    boostModel = BoostModel(inputImages, wholeTumorMasks, trainIdx);

    boostValidationScores = validateBoosting( ...
        validationSlices, boostModel, inputImages, wholeTumorMasks);
    [boostValidationPreds, ~, boostThreshold] = validateByThreshold( ...
        boostValidationScores, wholeTumorMasks, validationSlices, ...
        "Boosting");

    boostTestScores = validateBoosting( ...
        testSlices, boostModel, inputImages, wholeTumorMasks);
    boostTestPreds = boostTestScores >= boostThreshold;

    % U-Net: its threshold is selected on validation
    uNetCacheFile = fullfile(outputFolder, "unet_baseline.mat");
    [uNet, ~, unetValidationPreds, ~, unetThreshold] = U_Network( ...
        inputImages, segFiles, wholeTumorMasks, trainIdx, ...
        validationIdx, uNetCacheFile);

    % Compare raw/closed masks and validation-selected fusion on test
    runClosedFusionAnalysis(uNet, unetValidationPreds, unetThreshold, ...
        nnValidationPreds, nnTestPreds, ...
        boostValidationPreds, boostTestPreds, inputImages, ...
        wholeTumorMasks, validationIdx, testIdx, segFiles);
end

% Helpers
function perc = makePercent(decimal, pts)
    arguments 
        decimal, pts = 4
    end
     num = sprintf("%.*f", pts, decimal * 100);
    perc = string(num) + "%";
end

function pct = percentOrNaN(top, denom)
    if denom == 0
        pct = NaN;
    else    
        pct = 100 * top / denom;
    end
end

function roll = dice(pred, gt)
    denom = nnz(pred) + nnz(gt);
    if denom == 0
        roll = NaN;
    else
        roll = 2*nnz(pred & gt) / denom;
    end
     %percentOrNaN(2 * TP, 2 * TP + FP + FN);
end

function score = jaccard(pred, gt)
    denom = nnz(pred | gt);
    if denom == 0
        score = NaN;
    else
        score = nnz(pred & gt) / denom;
    end
   % = percentOrNaN(TP, TP + FP + FN);
end

function metrics = compareRegionPixels(pred, gt)
    assert(isequal(size(pred), size(gt)), ...
        "Prediction and ground truth must have the same size");
        % Only occurs when the dimensions are different between matrices
            % e.g. (240 x 240) vs (256 x 256) would be rejected

    pred = logical(pred);
    gt = logical(gt);
    
    TP = nnz(pred & gt);
    FP = nnz(pred & ~gt);
    FN = nnz(~pred & gt);
    TN = nnz(~pred & ~gt);
    
    % Of the pixels the model called (+), how many are in the GT?
    metrics.PredictedInsideGT_pct = percentOrNaN(TP, TP + FP);
    
    % Of the pixels the model called (+), how many are NOT in the GT?
    metrics.PredictedOutsideGT_pct = percentOrNaN(FP, TP + FP);
    
    % Of all GT_(+) Pixels, how many are RECOVERED in the model?
    metrics.GroundTruthRecovered_pct = percentOrNaN(TP, TP + FN);
    
    % Of all GT_(-) Pixels, how many are REJECTED in the model?
    metrics.BackgroundCorrect_pct  = percentOrNaN(TN, TN + FP);
    
    metrics.Dice = dice(pred, gt);
    metrics.Jaccard = jaccard(pred, gt); 
    metrics.TP = TP;
    metrics.FP = FP;
    metrics.FN = FN;
    metrics.TN = TN;
end
    
% NOTE: Supposed to be Metrics but shorter, 
% you can change it to 'Metrics' if it
% seems too much like Matrix mentally
function printMetrix(methodName, pred, gt)
    metrix = compareRegionPixels( ...
        pred, gt);
    
    fprintf("\n%s\n", methodName);
    fprintf("   Predicted Inside GT: %.2f%%\n", metrix.PredictedInsideGT_pct);
    fprintf("  Predicted Outside GT: %.2f%%\n", metrix.PredictedOutsideGT_pct);
    fprintf("Ground Truth Recovered: %.2f%%\n", metrix.GroundTruthRecovered_pct);
    fprintf("                  Dice: %s\n",     makePercent(metrix.Dice,2));
    fprintf("               Jaccard: %s\n",     makePercent(metrix.Jaccard,2));
end

% Path Router
function path = deterPath(name, Joseph_sharedPath, Aaron_sharedPath, Neeka_sharedPath)
    if(name == "Joseph")
        path = Joseph_sharedPath;
    elseif(name == "Aaron")
        path = Aaron_sharedPath;
    elseif(name == "Neeka")
        path = Neeka_sharedPath;
    else
        error("Could not find the path to project files");
        % path = "INVALID";
    end
end

% main % main % main % main % main % main % main % main % main % main % main % main % main  
                                % Lol JAN

% Loading in data, for now replace the below with your own local files
Joseph_sharedPath = "/Users/josephrexrode/UTDallas/2026-2027/MATH6346/FinalProject";
Aaron_sharedPath  = "C:/Users/aaron/Downloads/School/UTD/Medical Image Analysis/Final Project/";
Neeka_sharedPath  = "/Users/neeka/Documents/medicalimage analysis/finalproject";

name = "Joseph";
sharedPath = deterPath(name, Joseph_sharedPath, Aaron_sharedPath, Neeka_sharedPath);

% Extract file paths
parentFolder  = fullfile(sharedPath,   "BraTS2026subset");
outputFolder  = fullfile(sharedPath,   "2DSlices");
rawFolder     = fullfile(outputFolder, "raw");
segmentFolder = fullfile(outputFolder, "groundTruth");

PNGFiles = dir(fullfile(outputFolder, '*', '*00008*slice_053.png'));

% segFiles = dir(fullfile(segmentFolder, '*seg_slice_*.png'));

% Run slicing Algos: (Multi)Otsu
close all; % close images from previous runs 
clc;       % close Graphs from previous runs 
    % I think it clears the console too

niiFiles = loadFiles(parentFolder, outputFolder, rawFolder, segmentFolder);
convert2DFiles(niiFiles, rawFolder, segmentFolder);
runSliceTest(outputFolder);

% Machine Learning Training Set
% UNIVERSAL TRAINING AND VALIDATION DATA SETS 
[inputImages, groundTruthLabels, wholeTumorMasks, ...
    trainIdx, validationIdx, testIdx, segFiles] = ...
    patientTrainingData(parentFolder, outputFolder);

labelScheme = bratsMETLabelScheme();

for k = 1:height(labelScheme)
    fprintf("%s (ID %d): %d pixels\n", ...
        labelScheme.ClassName(k), labelScheme.ID(k), ...
        nnz(groundTruthLabels == labelScheme.ID(k)));
end

trainingMasks = wholeTumorMasks(:, :, trainIdx);
nnz(trainingMasks)/numel(trainingMasks);

% [stack, groundTruth] = loadSliceStack(outputFolder, "00002", 1);
% [stack, groundTruth] = loadSliceStack(outputFolder, "00008", 53);
% unique(groundTruth)
% nnz(groundTruth > 0)

% modelCacheFile = fullfile(outputFolder, "pixelNN.mat");
% runNeuralNetwork(inputImages, wholeTumorMasks, trainIdx, testIdx, validationIdx, modelCacheFile, segFiles, outputFolder);

% boosting
boostModel = BoostModel(inputImages, wholeTumorMasks, trainIdx);

validationSlices = find(validationIdx);
boostValidationScores = validateBoosting( ...
    validationSlices, boostModel, inputImages, wholeTumorMasks);

[boostValidationPreds, boostValidationDice, boostThreshold] = ...
    validateByThreshold(boostValidationScores, wholeTumorMasks, ...
    validationSlices, "Boosting");

boostTestGroundTruth = wholeTumorMasks(:, :, validationSlices);
printMetrix("Boosting Validation", boostValidationPreds, ...
    boostTestGroundTruth);


[validationPreds, validationDice, bestThreshold] = validateByThreshold( ...
    boostValidationScores, wholeTumorMasks, validationSlices...
);



testSlices = find(testIdx);
boostTestScores = validateBoosting( ...
    testSlices, boostModel, inputImages, wholeTumorMasks);
boostTestPreds = boostTestScores >= boostThreshold;
boostTestGroundTruth = wholeTumorMasks(:, :, testSlices);

printMetrix("Boosting test", boostTestPreds, boostTestGroundTruth);
% boosting Neural Network^^^


% fprintf("Boosting Validation Dice: %s\n",  makePercent(boostValidationDice));
% verifySlice("Boosting Prediction", inputImages, wholeTumorMasks, boostValidationPreds, validationIdx, segFiles);
% 

uNetCacheFile = fullfile(outputFolder, "unet_baseline.mat");

[uNet, trainInfo] = U_Network( ...
    inputImages, segFiles, wholeTumorMasks, trainIdx, ...
    validationIdx, uNetCacheFile);


edemaMask      = makeRegionMask(groundTruthLabels, labelScheme, "Edema");
tumorCoreMask  = makeRegionMask(groundTruthLabels, labelScheme, "TumorCore");
enhancingMask  = makeRegionMask(groundTruthLabels, labelScheme, "EnhancingTumor");
wholeTumorMask = makeRegionMask(groundTruthLabels, labelScheme, "WholeTumor");

% Combined Model
runSegmentionModels(inputImages, wholeTumorMasks, trainIdx, ...
    validationIdx, testIdx, segFiles, outputFolder);