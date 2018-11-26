function [data, image] = read_data(filename, is_RGB, visualize, save, save_name)
% read the text data file
%   [data, image] = read_data(filename, is_RGB) read the data file named 
%   filename. Return the data matrix with same shape as data in the file. 
%   If is_RGB is not true, the data will be regarded as Lab and convert to  
%   RGB format to visualise and save.
%
%   [data, image] = read_data(filename, is_RGB, visualize)  
%   If visualize is true, the data will be shown. Default value is false.
%
%   [data, image] = read_data(filename, is_RGB, visualize, save)  
%   If save is true, the image will be saved in an jpg image with same name
%   as the text filename. Default value is false.
%
%   [data, image] = read_data(filename, is_RGB, visualize, save, save_name)  
%   The image filename.
%
%   Example: [data, image] = read_data('1_noise.txt', true);
%   Example: [data, image] = read_data('cow.txt', false, true, true, 'segmented_cow.jpg');

    if nargin == 2
        visualize = false;
    elseif nargin == 3
        save = false;
    elseif nargin == 4
        save_name = strcat(filename(1 : length(filename)-4), '.jpg');
    else
        fprintf('Not enough input arguments\n You need to give "filename" and "is_RGB\n');
    end

    % read the text file
    data = dlmread(filename);
    
    % convert to RGB image
    [N, D] = size(data);     % data size
    cols = data(N, 1) + 1;   % number of columns
    rows = data(N, 2) + 1;   % number of rows
    img_data = data(:, 3 : D);
    image = reshape(img_data, [rows, cols, D-2]);
    if ~is_RGB
        cform = makecform('lab2srgb');
        image = applycform(image, cform);
    end
    
    % visualize image
    if visualize
        imshow(image, []);
    end
    
    % save
    if save
        imwrite(image, save_name);
    end

end