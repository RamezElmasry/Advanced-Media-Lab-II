clc;
clear all;
close all;
warning off;

% Apply object tracking on the video
tracks = GenerateObjectTrackers();
% Post_processing
% Reform and organize the data of the acquired trackers along with their detection
% coordinates and also remove false trackers(trackers with very low number of detections)
[final_ids, data] = reshape_data(tracks);
% Generate RGB videos according to the number of final unique ids
generate_outputs(final_ids, data);

function value = checker(box, data)
    if ismember(box, data)
        value = 1;
    else
        value = 0;
    end
end

function generate_outputs(final_ids, data)
    disp('Generating Outputs for each unique id')
    vid_count = 1;
    for i = 1:size(final_ids,2)
        % Set detetor settings same settings as done in the tracking
        % process.
        detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);
        blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 700);
        % Read the video
        vid = VideoReader('atrium_video3.mp4');
        % Output video settings
        vid_name = sprintf('FP_Output_%d', vid_count)
        img = readFrame(vid);
        v = VideoWriter(vid_name,'MPEG-4');
        v.FrameRate = 10;
        open(v);
        while hasFrame(vid)
            % Intilize a new black frame with every frame in the video
            out_frame = uint8(zeros(size(img,1),size(img,2),size(img,3)));
            % Read video frame
            frame = readFrame(vid);
            % Convert frame to gray scale
            gray = rgb2gray(frame);
            % Apply same detection parameters used in the tracking process
            mask = detector.step(gray);
            mask = imopen(mask, strel('rectangle', [3,3]));
            mask = imclose(mask, strel('rectangle', [15, 15]));
            mask = imfill(mask, 'holes');
            [~, ~, bboxes] = blobAnalyser.step(mask);
            % Start looping on the data acquired from the tracking process
            for j = 1:1:size(data,1)
                % check whether there is a detection or not
                if ~isempty(bboxes)
                    % check if any of the detections acquired in data is equal to any of
                    % the detected boxes in the current frame.
                    % This check is done for every unique id.
                    value = checker(data(j,2:end), bboxes);
                    if value == 1 && data(j,1:1) == final_ids(i)
                        % If found crop that object and put it on the black
                        % frame
                        cropped_person = imcrop(frame, data(j,2:end));
                        box = data(j,2:end);
                        out_frame(box(2):box(2)+box(4),box(1):box(1)+box(3),:) = cropped_person;
                        writeVideo(v,out_frame);
                        break
                    end
                end
            end         
        end
        close(v);
        vid_count = vid_count + 1;
    end
end

function [final_ids, data] = reshape_data(tracks)
    unique_ids = [];
    data = [];
    final_ids = [];
    count_occurences = [];
    [r,c] = size(tracks);
    i = 1;
    % Find unique ids
    while i <= c
        if length(num2str(tracks(i))) == 1 && ~ismember(tracks(i),unique_ids) && tracks(i) < 8
            unique_ids = [unique_ids, tracks(i)];
        end    
        i = i + 1;
    end
    % Reform obtained tracks into the shape of [id x y w h]
    j = 1;
    while j < c
        if length(num2str(tracks(j))) == 1 && length(num2str(tracks(j+1))) ~= 1 && ismember(tracks(j),unique_ids(:))           
            boxes = [tracks(j),tracks(j+1),tracks(j+2),tracks(j+3),tracks(j+4)];
            data = [data;boxes];   
        end
        
        if length(num2str(tracks(j))) == 1 && length(num2str(tracks(j+1))) == 1 && ismember(tracks(j),unique_ids(:)) && ismember(tracks(j+1),unique_ids(:))            
            boxes = [tracks(j),tracks(j+2),tracks(j+3),tracks(j+4),tracks(j+5)];
            data = [data;boxes];         
            boxes = [tracks(j+1),tracks(j+6),tracks(j+7),tracks(j+8),tracks(j+9)];
            data = [data;boxes];
            j = j + 1;
        end     
        j = j + 1;
    end
    % count the occurences of the unique ids in the data
    ids_col = data(:,1);
    count = 0;
    for d = 1:size(unique_ids,2)
        for k = 1:size(ids_col,1)
            if unique_ids(d) == ids_col(k)
                count  = count + 1;
            end
        end
        count_occurences = [count_occurences, count];
        count = 0;            
    end
    
    %Remove false trackers/detections (trackers with very low number of detections)
    total = 0;
    for c = 1:1:size(count_occurences,2)
        if count_occurences(c) < 15
            total = total + count_occurences(c);       
        else
            final_ids = [final_ids, unique_ids(c)];
        end
    end
    [p,~] = size(data);
    for row = 1:p-total
        if ~ismember(data(row,1), final_ids)
            data(row,:) = [];
        end
    end
end

function final_tracks = GenerateObjectTrackers()
disp('Tracking objects');
% Create System objects used for reading video, detecting moving objects,
% and displaying the results.
obj = setup();

tracks = initializeTracks(); % Create an empty array of tracks.
final_tracks = [];
nextId = 1; % ID of the next track

% Detect moving objects, and track them across video frames.
while hasFrame(obj.reader)
    frame = readFrame(obj.reader);
  
    [centroids, bboxes] = detectObjects(frame);
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTrackers();
    
    updateAssignedTracks();
    updateOtherTracks();
    deleteUneededTracks();
    createTracks();
end

function obj = setup()
        obj.reader = VideoReader('atrium_video3.mp4');

        obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);

        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 700);
end

function tracks = initializeTracks()
    
        tracks = struct(...
            'id', {}, ...
            'bbox', {}, ...
            'kalmanFilter', {}, ...
            'age', {}, ...
            'totalVisibleCount', {}, ...
            'consecutiveInvisibleCount', {});
end

function [centroids, bboxes, mask] = detectObjects(frame)
        gray = rgb2gray(frame);

        mask = obj.detector.step(gray);
        mask = imopen(mask, strel('rectangle', [3,3]));
        mask = imclose(mask, strel('rectangle', [15, 15]));
        mask = imfill(mask, 'holes');
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
end

function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;
            predictedCentroid = predict(tracks(i).kalmanFilter);
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
 end
function [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTrackers()

        nTracks = length(tracks);
        nDetections = size(centroids, 1);

        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end

        % Solve the assignment problem.
        costOfNonAssignment = 50;
        [assignments, unassignedTracks, unassignedDetections] = ...
            assignDetectionsToTracks(cost, costOfNonAssignment);
        
 end
function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);

            correct(tracks(trackIdx).kalmanFilter, centroid);
            tracks(trackIdx).bbox = bbox;

            tracks(trackIdx).age = tracks(trackIdx).age + 1;

            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
end
function updateOtherTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
end
function deleteUneededTracks()
        if isempty(tracks)
            return;
        end
         
        invisibleForTooLong = 80;
        ageThreshold = 150;
      
        final_tracks = [final_tracks, [tracks.id, tracks.bbox]];
             
        % Compute the fraction of the track's age for which it was visible.
        ages = [tracks(:).age];
        totalVisibleCounts = [tracks(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;

        % Find the indices of 'lost' tracks.
        lostInds = (ages < ageThreshold & visibility < 0.6) | ...
            [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
 
        % Delete lost tracks.
        tracks = tracks(~lostInds);
end
function createTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);

        for i = 1:size(centroids, 1)

            centroid = centroids(i,:);
            bbox = bboxes(i, :);

            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);

            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);

            tracks(end + 1) = newTrack;

            nextId = nextId + 1;
        end
end
end