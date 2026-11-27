function [uNet, trainInfo] = build_U_Network_Dice( ...
    inputImages, wholeTumorMasks, trainIdx, validationIdx, modelCacheFile, opts)
% BUILD_U_NETWORK_DICE  Train the 2-D U-Net with a Dice + cross-entropy loss.
%
% Drop-in replacement for build_U_Network (same inputs/outputs, plus options).
% Changes vs. build_U_Network:
%   * Loss = (1 - soft Dice) + cross-entropy, instead of cross-entropy alone.
%     Plain cross-entropy rewards "predict background" on a ~1%-tumor dataset,
%     which is why recall was low and the best threshold was far below 0.5.
%   * Tumor-containing training slices are repeated (TumorRepeat times); the
%     repeats beyond the first get a random left-right flip and a small
%     per-channel brightness change. Tumor-free slices are used once.
%   * Larger mini-batch (8), so each Dice estimate sees tumor pixels.
%   * Fixed random seed, so reruns are comparable.
%
% Example:
%   [uNet, info] = build_U_Network_Dice(inputImages, wholeTumorMasks, ...
%       trainIdx, validationIdx, fullfile(outputFolder, "unet_dice.mat"));
%   % optional: ..., Seed=2, EncoderDepth=4, NumFirstFilters=32)

    arguments
        inputImages
        wholeTumorMasks
        trainIdx
        validationIdx
        modelCacheFile (1,1) string
        opts.Seed (1,1) double = 1
        opts.TumorRepeat (1,1) double {mustBeInteger, mustBePositive} = 3
        opts.EncoderDepth (1,1) double = 3
        opts.NumFirstFilters (1,1) double = 16
        opts.MaxEpochs (1,1) double = 30
        opts.MiniBatchSize (1,1) double = 8
    end

    rng(opts.Seed);
    [H, W, C, ~] = size(inputImages);

    trainSlices = find(trainIdx);
    valSlices   = find(validationIdx);

    % ---- Training set: oversample + augment tumor slices -------------------
    hasTumor    = squeeze(any(wholeTumorMasks(:, :, trainSlices), [1 2]));
    tumorSlices = trainSlices(hasTumor);
    emptySlices = trainSlices(~hasTumor);

    nTotal = numel(tumorSlices) * opts.TumorRepeat + numel(emptySlices);
    XTrain = zeros(H, W, C, nTotal, "single");
    MTrain = false(H, W, nTotal);

    k = 0;
    for r = 1:opts.TumorRepeat
        for s = tumorSlices(:)'
            k = k + 1;
            [XTrain(:, :, :, k), MTrain(:, :, k)] = augmentSlice( ...
                inputImages(:, :, :, s), wholeTumorMasks(:, :, s), r > 1);
        end
    end
    for s = emptySlices(:)'
        k = k + 1;
        XTrain(:, :, :, k) = inputImages(:, :, :, s);
        MTrain(:, :, k)    = wholeTumorMasks(:, :, s);
    end

    order  = randperm(nTotal);
    XTrain = XTrain(:, :, :, order);
    MTrain = MTrain(:, :, order);

    fprintf("U-Net training set: %d tumor slices x %d, %d tumor-free slices -> %d total\n", ...
        numel(tumorSlices), opts.TumorRepeat, numel(emptySlices), nTotal);

    % One-hot targets, channel 1 = Background, channel 2 = Tumor
    % (the rest of the pipeline reads channel 2 as the tumor score)
    YTrain = oneHotMasks(MTrain);
    XVal   = single(inputImages(:, :, :, valSlices));
    YVal   = oneHotMasks(wholeTumorMasks(:, :, valSlices));

    % ---- Network and training ----------------------------------------------
    baselineNet = unet([H W C], 2, ...
        EncoderDepth = opts.EncoderDepth, ...
        NumFirstEncoderFilters = opts.NumFirstFilters);

    options = trainingOptions("adam", ...
        MaxEpochs = opts.MaxEpochs, ...
        MiniBatchSize = opts.MiniBatchSize, ...
        InitialLearnRate = 1e-3, ...
        Shuffle = "every-epoch", ...
        ValidationData = {XVal, YVal}, ...
        ValidationFrequency = 25, ...
        ValidationPatience = 10, ...
        OutputNetwork = "best-validation", ...
        Plots = "training-progress", ...
        Verbose = true);

    [uNet, trainInfo] = trainnet(XTrain, YTrain, baselineNet, ...
        @diceCrossEntropyLoss, options);

    save(modelCacheFile, "uNet", "trainInfo", "-v7.3");
    fprintf("U-Net (Dice loss) training complete and cached to:\n  %s\n", modelCacheFile);
end


function loss = diceCrossEntropyLoss(Y, T)
    % Y: softmax output (H x W x 2 x B), T: one-hot targets (same size).
    % Dice is computed over the whole mini-batch, which is more stable than
    % per-image Dice when some slices contain no tumor.
    smooth = 1;
    Yt = Y(:, :, 2, :);
    Tt = T(:, :, 2, :);
    inter    = sum(Yt .* Tt, "all");
    denom    = sum(Yt, "all") + sum(Tt, "all");
    softDice = (2 * inter + smooth) / (denom + smooth);
    loss = (1 - softDice) + crossentropy(Y, T, NormalizationFactor = "all-elements");
end


function Y = oneHotMasks(masks)
    [H, W, N] = size(masks);
    Y = zeros(H, W, 2, N, "single");
    Y(:, :, 1, :) = reshape(single(~masks), H, W, 1, N);
    Y(:, :, 2, :) = reshape(single(masks),  H, W, 1, N);
end


function [img, mask] = augmentSlice(img, mask, doAug)
    if ~doAug
        return;
    end
    if rand < 0.5                      % left-right flip (image and mask together)
        img  = flip(img, 2);
        mask = flip(mask, 2);
    end
    gain = 0.9 + 0.2 * rand(1, 1, size(img, 3), "single");   % per-channel brightness
    img  = min(img .* gain, 1);        % background stays exactly 0
end
