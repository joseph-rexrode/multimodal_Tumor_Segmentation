function [inputImages, groundTruthLabels, wholeTumorMasks, ...
    trainIdx, validationIdx, testIdx, segFiles] = buildSliceDataset( ...
    parentFolder, trainPatients, validationPatient, testPatient)
% BUILDSLICEDATASET  Build the 2-D slice dataset directly from the NIfTI volumes.
%
% Replaces the old pipeline (NIfTI -> per-slice mat2gray PNG -> reload).
% Differences from the PNG route:
%   * Each modality is normalized ONCE PER PATIENT VOLUME (not per slice), so
%     the same tissue keeps the same brightness on every slice.
%   * Labels are kept as the original integer values 0-3 (no 0/85/170/255).
%   * No intermediate files; values stay single-precision.
%
% Outputs have the same shapes/meaning as before, except:
%   inputImages       240 x 240 x 4 x N  SINGLE in [0,1]  (was uint8 0-255)
%                     channel order: t1n, t1c, t2w, t2f
%                     background (outside the brain) = 0; brain voxels are
%                     mapped to [0.01, 1] so "img > 0" still means "brain".
%   groundTruthLabels 240 x 240 x N     uint8, true BraTS labels 0..3
%   wholeTumorMasks   240 x 240 x N     logical, labels > 0
%   segFiles          struct array with only a .name field (compatibility
%                     shim so verifySlice keeps working; no file behind it)
%
% Slice order is patient by patient (in the order given, train patients
% first, then validation, then test), slices 1..D within each patient.

    modalities = ["t1n" "t1c" "t2w" "t2f"];   % channel order used everywhere

    trainPatients     = string(trainPatients(:));
    validationPatient = string(validationPatient);
    testPatient       = string(testPatient);
    allPatients = [trainPatients; validationPatient; testPatient];

    nPat = numel(allPatients);
    imagesByPatient = cell(nPat, 1);
    labelsByPatient = cell(nPat, 1);
    sliceCounts     = zeros(nPat, 1);

    for p = 1:nPat
        caseID     = "BraTS-MET-" + allPatients(p) + "-000";
        caseFolder = findCaseFolder(parentFolder, caseID);

        [imagesByPatient{p}, labelsByPatient{p}] = ...
            loadCase(caseFolder, caseID, modalities);

        sliceCounts(p) = size(labelsByPatient{p}, 3);
        fprintf("Loaded %s: %d slices\n", caseID, sliceCounts(p));
    end

    inputImages       = cat(4, imagesByPatient{:});
    groundTruthLabels = cat(3, labelsByPatient{:});
    wholeTumorMasks   = groundTruthLabels > 0;   % labels 1, 2, 3 are all "WholeTumor"

    % Which patient does each slice belong to?
    patientOfSlice = repelem(allPatients, sliceCounts);
    trainIdx       = ismember(patientOfSlice, trainPatients);
    validationIdx  = patientOfSlice == validationPatient;
    testIdx        = patientOfSlice == testPatient;

    % Compatibility shim: names that look like the old segmentation PNG names
    nTotal = numel(patientOfSlice);
    names  = cell(nTotal, 1);
    k = 0;
    for p = 1:nPat
        for s = 1:sliceCounts(p)
            k = k + 1;
            % char(...) on purpose: dir() returned char names, and
            % verifySlice's contains({segFiles.name}, ...) needs char, not string.
            names{k} = char(sprintf( ...
                "BraTS-MET-%s-000_BraTS-MET-%s-000-seg_slice_%03d.png", ...
                allPatients(p), allPatients(p), s));
        end
    end
    segFiles = struct("name", names);

    fprintf("Total labeled slices: %d\n", nTotal);
    fprintf("Training Slices: %d\n",   nnz(trainIdx));
    fprintf("Validation Slices: %d\n", nnz(validationIdx));
    fprintf("Test Slices: %d\n",       nnz(testIdx));
end


function caseFolder = findCaseFolder(parentFolder, caseID)
    hits = dir(fullfile(parentFolder, "**", caseID + "-seg.nii.gz"));
    if isempty(hits)
        error("Could not find %s-seg.nii.gz under %s", caseID, parentFolder);
    end
    caseFolder = hits(1).folder;
end


function [images, labels] = loadCase(caseFolder, caseID, modalities)
    % Ground truth: keep the original integer labels
    seg    = niftiread(fullfile(caseFolder, caseID + "-seg.nii.gz"));
    labels = uint8(seg);

    unexpected = setdiff(unique(labels(:)), uint8(0:3));
    if ~isempty(unexpected)
        error("%s: unexpected label values %s (expected only 0-3).", ...
            caseID, mat2str(double(unexpected(:)')));
    end

    % Raw modality volumes
    vols = cell(1, numel(modalities));
    for m = 1:numel(modalities)
        vols{m} = single(niftiread(fullfile( ...
            caseFolder, caseID + "-" + modalities(m) + ".nii.gz")));
        if ~isequal(size(vols{m}), size(labels))
            error("%s: %s volume size %s does not match segmentation size %s.", ...
                caseID, modalities(m), mat2str(size(vols{m})), mat2str(size(labels)));
        end
    end

    % BraTS volumes are skull-stripped, so non-zero voxels = brain
    brain = any(cat(4, vols{:}) > 0, 4);

    [H, W, D] = size(labels);
    images = zeros(H, W, numel(modalities), D, "single");
    for m = 1:numel(modalities)
        norm3D = normalizeVolume(vols{m}, brain);
        images(:, :, m, :) = reshape(norm3D, H, W, 1, D);
    end
end


function out = normalizeVolume(vol, brain)
    % Per-volume, per-modality normalization using brain voxels only:
    %   1. clip to the 1st-99.5th percentile (tames bright vessels / outliers)
    %   2. rescale to [0.01, 1]
    % Everything outside the brain stays exactly 0.
    vals = vol(brain);
    lo = prctile(vals, 1);
    hi = prctile(vals, 99.5);

    out = zeros(size(vol), "single");
    if hi <= lo
        warning("Degenerate intensity range in a volume; leaving it at 0.01 inside the brain.");
        out(brain) = 0.01;
        return;
    end

    clipped = min(max(vol(brain), lo), hi);
    out(brain) = 0.01 + 0.99 * (clipped - lo) / (hi - lo);
end
