function [J, H, Jmetric, Jener, sig_vec] = obj_RR(Y, L, Dt, Q, calDim)
global weight epsilon_inv
% Y: data (num of data X dim)
% L: Laplace-Beltrami operator
% Dt: normalized diagonal matrix
% Q: quadratic form to calculate energy
% calDim: dimension for calcuate distortion (may be different from dim, and corresponds to intrinsic dim in RR paper)
if nargin < 5
    calDim = [];
    if nargin < 4
       Q = []; 
    end
end
N = size(L,1);
dim = size(Y,2);

% calculate components of metric inv and deriv of metric inv
[H] = estimateMetricInv(Y, L, false);
di = zeros(N,dim);

% calculate obj ftn
J = 0;
sig_vec = zeros(1,N);
for i = 1:N
    [~, Di] = eig(H(:,:,i));
    di(i,:) = diag(Di)';
    if ~isempty(di(i,di(i,:)<0))
        error('not psd')
    end
    if isempty(calDim)
        temp = di(i,:) - ones(1,dim);
        [disqtemp] = sort(temp.*temp, 2, 'descend');
        J = J + disqtemp(1) * Dt(i,i);
        sig_vec(i) = disqtemp(1);
    else
        [ditemp] = sort(di(i,:), 2, 'descend');
        temp = [ditemp(1:calDim) - 1, ditemp(calDim+1:end) * epsilon_inv];
        [disqtemp] = sort(temp.*temp, 2, 'descend');
        J = J + disqtemp(1) * Dt(i,i);
        sig_vec(i) = temp(1).*temp(1);
    end
end
Jmetric = J;
Jener = 0;
if (weight > 0 && ~isempty(Q))
    Jener = trace(Y'*Q*Y);
    J = J + weight * Jener;
end