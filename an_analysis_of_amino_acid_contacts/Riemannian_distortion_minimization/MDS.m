function [Y, V, D, T] = MDS(data, dim, distMat_init)
% data: input
% dim: embedding dimension
% distMat_init: ambient space (non-squared) distance matrix
% Y: output embedding
% V: eigenvector
% D: eigenvalue

if isempty(data)
    N = size(distMat_init,1);
else
    N = size(data,1);
end
if isempty(distMat_init)
    distMat_init = getPairwiseDist(data);
end
S = distMat_init.*distMat_init;
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
