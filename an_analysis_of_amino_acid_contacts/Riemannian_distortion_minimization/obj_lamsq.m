function [J, H, Jmetric, Jener, sig_vec] = obj_lamsq(Y, L, Dt, Q, calDim)
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
%     if ~isempty(di(i,di(i,:)<0))
%         error('not psd')
%     end
    if isempty(calDim) || calDim == dim
        sig_vec(i) = (0.5*trace(H(:,:,i)^2) - trace(H(:,:,i)));
        J = J + (0.5*trace(H(:,:,i)^2) - trace(H(:,:,i))) * Dt(i,i);
%         g2 = g2 + 2*(Di(idx,idx)-1)*reshape((diag(L(i,:))*Y - I(:,i)*L(i,:)*Y - L(i,:)'*Y(i,:))*Vi(:,idx)*Vi(:,idx)', dim*N,1) * Dt(i,i);
    else
        di(i,:) = eig(H(:,:,i))';
        if ~isempty(di(i,di(i,:)<0))
            if length(di(i,di(i,:)<0)) > dim - calDim
                error('not psd')
            end
        end
        [ditemp] = sort(di(i,:), 2, 'descend');
        objdi = ditemp(1:calDim) - 1;
        penalty = max(ditemp(calDim+1:end) - epsilon, 0);
        weight_penalty_sqrt = sqrt(weight_penalty);
        obji = [objdi, weight_penalty_sqrt * penalty];
        sig_vec(i) = sum(objdi.*objdi, 2);
        J = J + sum(obji.*obji, 2) * Dt(i,i);
    end
end
Jmetric = J;
Jener = 0;
if (weight > 0 && ~isempty(Q))
    Jener = trace(Y'*Q*Y);
    J = J + weight * Jener;
end




