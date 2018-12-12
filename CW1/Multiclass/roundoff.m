%Find the highest value and change it to 1
%The lower values will be changed to 0
function [return_matrix] = roundoff(y)
    [row, column] = size(y);
    return_matrix = zeros( row, column);
    
    for i=1:column
        max = -1;
        max_pos = 0;
        for j = 1: row
            if y( j, i) > max
                max_pos = j;
                max = y( j, i);
            end
        end
        return_matrix( max_pos, i) = 1;
    end
end