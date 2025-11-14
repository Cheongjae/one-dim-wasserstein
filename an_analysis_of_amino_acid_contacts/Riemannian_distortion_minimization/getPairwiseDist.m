function [distanceMat, idx] = getPairwiseDist(Y, k, idx)
if nargin < 3
    idx = [];
    if nargin < 2
        k = [];
    end
end
N = size(Y,1);
if isempty(k) && isempty(idx)
    % calculate distance for all the pairs
    distanceMat = zeros(N,N);
    for i = 1:N
        for j = i+1:N
            distanceMat(i,j) = norm(Y(i,:) - Y(j,:));
            distanceMat(j,i) = distanceMat(i,j);
        end
    end
elseif (~isempty(k) && size(idx,2) == k) || size(idx, 1) == N
    distanceMat = zeros(N,size(idx,2));
    for i = 1:size(idx,2)
        dYi = Y - Y(idx(:,i),:);
        distanceMat(:,i) = sqrt(sum(dYi.*dYi,2));
    end
elseif ~isempty(k)
    % k-NN search
    [idx] = knnsearch(Y, Y, 'k', k+1);
    idx = idx(:,2:end);
    distanceMat = zeros(N,size(idx,2));
    for i = 1:size(idx,2)
        dYi = Y - Y(idx(:,i),:);
        distanceMat(:,i) = sqrt(sum(dYi.*dYi,2));
    end
else
    distanceMat = [];
    idx = [];
end