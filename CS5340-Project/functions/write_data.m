function write_data(data, filename)
% write the matrix into a text file
%   write_data(data, filename) write 2d matrix data into a text file named
%   filename.
%
%   Example: write_data(data, 'cow.txt');

    dlmwrite(filename, data, 'delimiter', ' ', 'precision', 8);

end