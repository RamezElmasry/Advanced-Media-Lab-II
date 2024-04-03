clc;
clear all;

img = imread('4.jpg');
k = 4;
C = [100 100 100 ;128 128 128 ;175 175 175 ;255 255 255];
%out1 = average_filter(img,5);
% pre-processing (cropping the road out of the image).
out = img(120:286,1:390,:);
% Apply K-means Clustering on the image with intial centers and K = 4.
k_out = imgkmeansSegmentation(out,k,C);
% Post-processing (apply median filter to remove the noise at the background).
filtered_img = median_filter(k_out,7);
imshow(filtered_img);

function img = imgkmeansSegmentation(img,k,C)
    subplot(1,2,1);
    imshow(img);
    title('Original Image');
    R = img(:, :, 1);
    G = img(:, :, 2);
    B = img(:, :, 3);
    data = double([R(:), G(:), B(:)]);
    [m, n, sumd] = kmeans(data,k,'Start',C);
    m = reshape(m,size(img,1),size(img,2));
    n;
    [rows,cols] = size(m);
    n = n/255;
    %clusteredImage_rgb = label2rgb(m,n);
    %subplot(1,3,2);
    %imshow(clusteredImage_rgb);
    %title(['RGB with ', 'k = ', num2str(k)]); 
    img = rgb2gray(img);
    for i = 1:1:rows
        for j = 1:1:cols
           if m(i,j) == 1
               img(i,j) = 0;
           end
           if m(i,j) == 2
               img(i,j) = 255;
           end
           if m(i,j) == 3
               img(i,j) = 0;
           end
           if m(i,j) == 4
               img(i,j) = 0;
           end
        end
    end
    subplot(1,2,2);
    imshow(uint8(img));
    title(['Gray with ', 'k = ', num2str(k)]);      
end
function [img] = median_filter(img, n)
    [M, N] = size(img);
    kernel_size = n;
    k = zeros(kernel_size);  %k is the kernel used. 
    start = kernel_size - floor(kernel_size*0.5);
    for x = start:1:M-floor(kernel_size*0.5)
        for y = start:1:N-floor(kernel_size*0.5)
            x1 = x - (floor(kernel_size*0.5));
            y1 = y - (floor(kernel_size*0.5));

    %specifying image pixels to the kernel
            for p = 1:1:kernel_size
                for q = 1:1:kernel_size
                    k(p,q) = img(x1+p-1,y1+q-1);
                end    
            end
        d=reshape(k,1,[]);  %k values into an array d 
        [r,c]=size(d);

        Median=median(sort(d));

       img(x,y)=Median;    
        end
    end
end
