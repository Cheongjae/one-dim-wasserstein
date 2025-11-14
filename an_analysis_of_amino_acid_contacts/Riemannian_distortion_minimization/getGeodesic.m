function [length, idxset, G] = getGeodesic(idx1, idx2, data, k, metric, distMat_init, G)
% find geodesic between two data points using graph structure
if nargin < 7
    G = [];
    if nargin < 6
        distMat_init = [];
        if nargin < 5
            metric = [];
        end
    end
end
if isempty(G)
    %% k-NN search
    if isempty(distMat_init)
        N = size(data,1);
        [idx, dist] = knnsearch(data, data, 'k', k+1);
        idx = idx(:,2:end);
        dist = dist(:,2:end);
    else
        N = size(distMat_init,1);
        [dist, idx] = sort(distMat_init, 2);
        idx = idx(:,2:k+1);
        dist = dist(:,2:k+1);
    end
    %% estimate graph distance
    distMat = zeros(N,N);
    % initialize
    for i = 1:N
        if isempty(metric)
            distMat(i,idx(i,:)) = dist(i,:);
            distMat(idx(i,:),i) = dist(i,:)';
        else
            for j = 1:k
                dx = data(i,:) - data(idx(i,j),:);
                tempdist = 0.5*(sqrt(dx*metric(:,:,i)*dx') + sqrt(dx*metric(:,:,idx(i,j))*dx'));
                distMat(i,idx(i,j)) = tempdist;
                distMat(idx(i,j),i) = tempdist;
            end
        end
    end
    
    G = graph(distMat);
end
[idxset, length] = shortestpath(G, idx1, idx2);
