function [Y, G, V, D, T, S] = isomap(data, k, dim, S, distMat_init)
% data: input
% k: number of nearest neighbor
% dim: embedding dimension
% S: path distance matrix
% distMat_init: ambient space (non-squared) distance matrix
% Y: output embedding
% G: graph
% V: eigenvector
% D: eigenvalue
G = [];
if nargin < 5
    distMat_init = [];
    if nargin < 4
        S = [];
    end
end
if isempty(data)
    N = size(distMat_init,1);
else
    N = size(data,1);
end
if isempty(S)
    S = zeros(N,N);
    for i = 1:N
        for j = i:N
            if i == 1 && j == 1
                [S(i,j), ~, G] = getGeodesic(i,j,data, k, [], distMat_init);
            else
                [~, S(i,j)] = shortestpath(G, i, j);
                S(j,i) = S(i,j);
            end
        end
    end
    S = S.*S;
end

H = eye(N) - 1/N;
T = -0.5*H*S*H;
if isempty(find(isnan(T),1)) && isempty(find(isinf(T),1))
    [V, D] = eig(T);
    d = diag(D);
    [~, idx] = sort(d, 'descend');
    V_ = V(:,idx);
    D_ = diag(d(idx));
    Y = V_(:,1:dim)*sqrt(D_(1:dim,1:dim));
else
    T = [];
    Y = [];
    V = [];
    D = [];
end