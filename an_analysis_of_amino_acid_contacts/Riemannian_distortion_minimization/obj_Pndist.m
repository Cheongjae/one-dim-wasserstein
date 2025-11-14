function [J, H, di, Jmetric, Jener, sig_vec] = obj_Pndist(Y, L, Dt, Q, calDim)
global weight epsilon weight_penalty
% L: Laplace-Beltrami operator
% Y: data (number of data X dim)
% Dt: normalized diagonal matrix
% Q: quadratic form to calculate energy
% calDim: dimension for calcuate distortion (may be different from dim)
if nargin < 5
    calDim = [];
    if nargin < 4
        Q = [];
    end
end

dim = size(Y,2);
N = size(Y,1);

% calculate components of metric inv and deriv of metric inv
[H] = estimateMetricInv(Y, L, false);
di = zeros(N,dim);

if ~isempty(calDim) && (calDim ~= dim)
    epsilon_inv = 1/epsilon;
end

J = 0;
sig_vec = zeros(1,N);
for i = 1:N
    di(i,:) = eig(H(:,:,i))';
    if ~isempty(di(i,di(i,:)<0))
        error('not psd')
    end
    if isempty(calDim) || calDim == dim
        obji = log(di(i,:));
        sig_vec(i) = sum(obji.*obji,2);
    else
        [ditemp] = sort(di(i,:), 2, 'descend');
        logdi = log(ditemp(1:calDim));
        penalty = max(log(ditemp(calDim+1:end)*epsilon_inv), 0);
        weight_penalty_sqrt = sqrt(weight_penalty);
        obji = [logdi, weight_penalty_sqrt * penalty];
        sig_vec(i) = sum(logdi.*logdi,2);
    end
    J = J + sum(obji.*obji,2) * Dt(i,i);
end
Jmetric = J;
Jener = 0;
if (weight > 0 && ~isempty(Q))
    Jener = trace(Y'*Q*Y);
    J = J + weight * Jener;
end