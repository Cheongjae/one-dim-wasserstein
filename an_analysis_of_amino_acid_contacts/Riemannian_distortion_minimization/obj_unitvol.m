function [J, H, Jmetric, Jener, sig_vec] = obj_unitvol(Y, L, Dt, Q, calDim)
global weight epsilon weight_penalty
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

% calculate obj ftn and grad
J = 0;
sig_vec = zeros(1,N);
for i = 1:N
%     [~, Di] = eig(H(:,:,i));
%     di(i,:) = diag(Di)';
    if ~isempty(di(i,di(i,:)<0))
        error('not psd')
    end
    if isempty(calDim) || calDim == dim
        detH = det(H(:,:,i));
        logdetH = log(detH);
        J = J + logdetH^2 * Dt(i,i);
        sig_vec(i) = logdetH^2;
    else
        di(i,:) = eig(H(:,:,i))';
        if ~isempty(di(i,di(i,:)<0))
            error('not psd')
        end
        [ditemp] = sort(di(i,:), 2, 'descend');
        penalty = max(ditemp(calDim+1:end) - epsilon, 0);
        logdetH_calDim = sum(log(ditemp(1:calDim)), 2);
        J = J + (logdetH_calDim^2 + weight_penalty * sum(penalty.*penalty, 2)) * Dt(i,i);
        sig_vec(i) = logdetH_calDim^2;
    end
end
Jmetric = J;
Jener = 0;
if (weight > 0 && ~isempty(Q))
    Jener = trace(Y'*Q*Y);
    J = J + weight * Jener;
end




