function [pairAngle, idx] = getNNPairwiseAngle(data, k, idx)
% calculate pairwise angle of NN vectors (NN point - data point) at each point
if nargin < 3
    idx = [];
end
N = size(data,1);
if isempty(idx) || k ~= size(idx,2)
    % k-NN search
    [idx] = knnsearch(data, data, 'k', k+1);
    idx = idx(:,2:end);
end
pairAngle = zeros(N, k*(k-1)/2);

for i = 1:N
    num = 0;
    for m = 1:k
        for n = m+1:k
            num = num + 1;
            vec1 = data(idx(i,m), :) - data(i,:);
            vec2 = data(idx(i,n), :) - data(i,:);
            temp = vec1*vec2'/norm(vec1)/norm(vec2);
            if temp > 1
                temp = 1;
            elseif temp < -1
                temp = -1;
            end
            pairAngle(i,num) = acos(temp);
        end
    end
end