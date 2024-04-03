clc;
clear all;


%Initialize a video reader and video writer objects
vid = VideoReader('mp2_small_video.mp4');
v = VideoWriter('mp2_output');
open(v);
while hasFrame(vid)
    % read frame from video
    vidFrame = readFrame(vid);
    % preprocess- convert rgb frame into binary
    frame_bin = imbinarize(rgb2gray(vidFrame));
    % Detect Circles in the frame/ and extract centers and radii information
    [centers,radii] = imfindcircles(frame_bin,[10,40]);
    [m,n] = size(centers);
    [r,c] = size(radii);
    % Draw a bounding box around the detected circles
    for i = 1:m
        vidFrame = insertShape(vidFrame,'Rectangle',[centers(i,1)-(2*radii(i)),centers(i,2)-(2*radii(i)),40,40],'Color','red'); 
    end
    % save the frame into the video writer.
    writeVideo(v,vidFrame);
end
close(v);
