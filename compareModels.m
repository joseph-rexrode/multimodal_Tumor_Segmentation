function results = compareModels(inputImages, wholeTumorMasks, ...
    validationIdx, testIdx, nnCacheFile, unetCacheFile)
% COMPAREMODELS  One like-for-like table: pixel NN, U-Net, and their average.
%
% For every model the threshold is chosen on the VALIDATION patient only,
% frozen, then applied to the TEST patient. Reports for each model:
%   ValDice, TestDice, TestPrecision, TestRecall ("Ground Truth Recovered"),
%   and TestDiceTumorSlices (Dice using only slices that contain tumor).
%
% Self-contained: needs only the cached models (pixelNN.mat, unet file).
% Boosting is intentionally left out until its constant-output problem is fixed.

    valSlices  = find(validationIdx);
    testSlices = find(testIdx);
    valGT  = wholeTumorMasks(:, :, valSlices);
    testGT = wholeTumorMasks(:, :, testSlices);

    tmp = load(nnCacheFile, "nnModel");     nnModel = tmp.nnModel;
    tmp = load(unetCacheFile, "uNet");      uNet    = tmp.uNet;

    valScores  = cell(1, 3);
    testScores = cell(1, 3);
    valScores{1}  = scoreNN(nnModel, inputImages, valSlices);
    testScores{1} = scoreNN(nnModel, inputImages, testSlices);
    valScores{2}  = scoreUNet(uNet, inputImages, valSlices);
    testScores{2} = scoreUNet(uNet, inputImages, testSlices);
    valScores{3}  = (valScores{1}  + valScores{2})  / 2;   % average probabilities
    testScores{3} = (testScores{1} + testScores{2}) / 2;

    modelNames = ["Pixel NN"; "U-Net"; "Average (NN + U-Net)"];
    thresholds = 0.05:0.05:0.95;

    Threshold = zeros(3, 1);  ValDice = zeros(3, 1);  TestDice = zeros(3, 1);
    TestPrecision = zeros(3, 1);  TestRecall = zeros(3, 1);
    TestDiceTumorSlices = zeros(3, 1);

    for m = 1:3
        bestDice = -1;  bestT = 0.5;
        for t = thresholds
            d = metricsOf(valScores{m} >= t, valGT).dice;
            if d > bestDice
                bestDice = d;  bestT = t;
            end
        end
        vm = metricsOf(valScores{m}  >= bestT, valGT);
        tm = metricsOf(testScores{m} >= bestT, testGT);

        Threshold(m) = bestT;
        ValDice(m) = 100 * vm.dice;
        TestDice(m) = 100 * tm.dice;
        TestPrecision(m) = 100 * tm.precision;
        TestRecall(m) = 100 * tm.recall;
        TestDiceTumorSlices(m) = 100 * tm.diceTumorSlices;
    end

    results = table(modelNames, Threshold, ValDice, TestDice, ...
        TestPrecision, TestRecall, TestDiceTumorSlices, ...
        VariableNames = ["Model" "Threshold" "ValDice" "TestDice" ...
        "TestPrecision" "TestRecall" "TestDiceTumorSlices"]);

    fprintf("\nWhole-tumor segmentation, all values in %% (threshold chosen on validation):\n");
    disp(results);
end


function scores = scoreNN(nnModel, inputImages, slices)
    [H, W, C, ~] = size(inputImages);
    scores = zeros(H, W, numel(slices), "single");
    for i = 1:numel(slices)
        img    = inputImages(:, :, :, slices(i));
        brain  = find(any(img > 0, 3));
        if isempty(brain)
            continue;
        end
        feats = reshape(single(img), [], C);          % one row per pixel
        s = minibatchpredict(nnModel, feats(brain, :), MiniBatchSize = 2048);
        full = zeros(H * W, 1, "single");
        full(brain) = s(:, 2);                        % column 2 = Tumor
        scores(:, :, i) = reshape(full, H, W);
    end
end


function scores = scoreUNet(uNet, inputImages, slices)
    [H, W, ~, ~] = size(inputImages);
    X   = single(inputImages(:, :, :, slices));
    out = minibatchpredict(uNet, X, MiniBatchSize = 8);   % H x W x 2 x N
    scores = reshape(out(:, :, 2, :), H, W, numel(slices));  % channel 2 = Tumor
end


function m = metricsOf(pred, gt)
    tp = nnz(pred & gt);  fp = nnz(pred & ~gt);  fn = nnz(~pred & gt);
    m.dice      = safeDiv(2 * tp, 2 * tp + fp + fn);
    m.precision = safeDiv(tp, tp + fp);
    m.recall    = safeDiv(tp, tp + fn);

    has = squeeze(any(gt, [1 2]));
    m.diceTumorSlices = safeDiv( ...
        2 * nnz(pred(:, :, has) & gt(:, :, has)), ...
        nnz(pred(:, :, has)) + nnz(gt(:, :, has)));
end


function r = safeDiv(a, b)
    if b == 0
        r = NaN;
    else
        r = a / b;
    end
end
