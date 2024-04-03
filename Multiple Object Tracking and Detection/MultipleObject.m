clc
clear all
close all
warning off

tracks = MotionBasedMultiObjectTrackingExample();
% Post_processing
% Reform and organize the data of the trackers along with their detection
% coordinates and also remove false trackers(trackers with low number of detections)
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
    vid_count = 1;
    for i = 1:size(final_ids,2)
        detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);
        blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 700);
        vid = VideoReader('atrium_video3.mp4');
        vid_name = sprintf('FP_Output_%d', vid_count)
        img = readFrame(vid);
        v = VideoWriter(vid_name,'MPEG-4');
        v.FrameRate = 10;
        open(v);
        while hasFrame(vid)
            out_frame = uint8(zeros(size(img,1),size(img,2),size(img,3)));
            frame = readFrame(vid);
            gray = rgb2gray(frame);
            mask = detector.step(gray);
            mask = imopen(mask, strel('rectangle', [3,3]));
            mask = imclose(mask, strel('rectangle', [15, 15]));
            mask = imfill(mask, 'holes');
            [~, ~, bboxes] = blobAnalyser.step(mask); 
            for j = 1:1:size(data,1)
                if ~isempty(bboxes)
                    value = checker(data(j,2:end), bboxes);
                    if value == 1 && data(j,1:1) == final_ids(i)
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
    while i <= c
        if length(num2str(tracks(i))) == 1 && ~ismember(tracks(i),unique_ids) && tracks(i) < 8
            unique_ids = [unique_ids, tracks(i)];
        end    
        i = i + 1;
    end

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
    
    ids_col = data(:,1);
    %ids_col = reshape(ids_col, [1,501]);
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
    total = 0;
    %remove false trackers/detections
    for c = 1:1:size(count_occurences,2)
        count_occurences(c)
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


function final_tracks = MotionBasedMultiObjectTrackingExample()
% Create System objects used for reading video, detecting moving objects,
% and displaying the results.
obj = setupSystemObjects();

tracks = initializeTracks(); % Create an empty array of tracks.
final_tracks = [];
nextId = 1; % ID of the next track

% Detect moving objects, and track them across video frames.
while hasFrame(obj.reader)
    frame = readFrame(obj.reader);
  
    [centroids, bboxes, mask] = detectObjects(frame);
    predictNewLocationsOfTracks();
    [assignments, unassignedTracks, unassignedDetections] = ...
        detectionToTrackAssignment();
    
    updateAssignedTracks();
    updateUnassignedTracks();
    deleteLostTracks();
    createNewTracks();
    displayTrackingResults();
end

function obj = setupSystemObjects()
        % Initialize Video I/O
        % Create objects for reading a video from a file, drawing the tracked
        % objects in each frame, and playing the video.

        % Create a video reader.
        obj.reader = VideoReader('atrium_video3.mp4');

        % Create two video players, one to display the video,
        % and one to display the foreground mask.
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 200, 700, 400]);
        obj.videoPlayer = vision.VideoPlayer('Position', [20, 200, 700, 400]);

        % Create System objects for foreground detection and blob analysis

        % The foreground detector is used to segment moving objects from
        % the background. It outputs a binary mask, where the pixel value
        % of 1 corresponds to the foreground and the value of 0 corresponds
        % to the background.

        obj.detector = vision.ForegroundDetector('NumGaussians', 3, ...
            'NumTrainingFrames', 40, 'MinimumBackgroundRatio', 0.7);

        % Connected groups of foreground pixels are likely to correspond to moving
        % objects.  The blob analysis System object is used to find such groups
        % (called 'blobs' or 'connected components'), and compute their
        % characteristics, such as area, centroid, and the bounding box.

        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
            'AreaOutputPort', true, 'CentroidOutputPort', true, ...
            'MinimumBlobArea', 700);
end

function tracks = initializeTracks()
        % create an empty array of tracks
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
        % Detect foreground.
        mask = obj.detector.step(gray);

        % Apply morphological operations to remove noise and fill in holes.
        mask = imopen(mask, strel('rectangle', [3,3]));
        mask = imclose(mask, strel('rectangle', [15, 15]));
        mask = imfill(mask, 'holes');

        % Perform blob analysis to find connected components.
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
end

 function predictNewLocationsOfTracks()
        for i = 1:length(tracks)
            bbox = tracks(i).bbox;

            % Predict the current location of the track.
            predictedCentroid = predict(tracks(i).kalmanFilter);

            % Shift the bounding box so that its center is at
            % the predicted location.
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
 end
 function [assignments, unassignedTracks, unassignedDetections] = ...
            detectionToTrackAssignment()

        nTracks = length(tracks);
        nDetections = size(centroids, 1);

        % Compute the cost of assigning each detection to each track.
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
        end

        % Solve the assignment problem.
        costOfNonAssignment = 20;
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

            % Correct the estimate of the object's location
            % using the new detection.
            correct(tracks(trackIdx).kalmanFilter, centroid);

            % Replace predicted bounding box with detected
            % bounding box.
            tracks(trackIdx).bbox = bbox;

            % Update track's age.
            tracks(trackIdx).age = tracks(trackIdx).age + 1;

            % Update visibility.
            tracks(trackIdx).totalVisibleCount = ...
                tracks(trackIdx).totalVisibleCount + 1;
            tracks(trackIdx).consecutiveInvisibleCount = 0;
        end
end
function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;
            tracks(ind).consecutiveInvisibleCount = ...
                tracks(ind).consecutiveInvisibleCount + 1;
        end
end
function deleteLostTracks()
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
function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);

        for i = 1:size(centroids, 1)

            centroid = centroids(i,:);
            bbox = bboxes(i, :);

            % Create a Kalman filter object.
            kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
                centroid, [200, 50], [100, 25], 100);

            % Create a new track.
            newTrack = struct(...
                'id', nextId, ...
                'bbox', bbox, ...
                'kalmanFilter', kalmanFilter, ...
                'age', 1, ...
                'totalVisibleCount', 1, ...
                'consecutiveInvisibleCount', 0);

            % Add it to the array of tracks.
            tracks(end + 1) = newTrack;

            % Increment the next id.
            nextId = nextId + 1;
        end
end
function displayTrackingResults()
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;

        minVisibleCount = 8;
        if ~isempty(tracks)

            % Noisy detections tend to result in short-lived tracks.
            % Only display tracks that have been visible for more than
            % a minimum number of frames.
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);

            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                bboxes = cat(1, reliableTracks.bbox);

                % Get ids.
                ids = int32([reliableTracks(:).id]);

                % Create labels for objects indicating the ones for
                % which we display the predicted rather than the actual
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);

                % Draw the objects on the frame.
                frame = insertObjectAnnotation(frame, 'rectangle', ...
                    bboxes, labels);

                % Draw the objects on the mask.
                mask = insertObjectAnnotation(mask, 'rectangle', ...
                    bboxes, labels);
            end
        end

        % Display the mask and the frame.
        obj.maskPlayer.step(mask);
        obj.videoPlayer.step(frame);
 end
end