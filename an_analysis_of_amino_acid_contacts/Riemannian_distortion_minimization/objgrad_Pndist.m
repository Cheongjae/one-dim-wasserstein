function [J, g, H, dHdz, Jmetric, Jener] = objgrad_Pndist(Y, L, Dt, Q, calDim)
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
g = zeros(size(Y,1)*size(Y,2), 1);
N = size(L,1);
dim = size(Y,2);

% calculate components of metric inv and deriv of metric inv
[H, dHdz] = estimateMetricInv(Y, L);
di = zeros(N,dim);

if ~isempty(calDim) && (calDim ~= dim)
    epsilon_inv = 1/epsilon;
end
% calculate obj ftn and grad
J = 0;
for i = 1:N
    [Vi, Di] = eig(H(:,:,i));
    di(i,:) = diag(Di)';
    if ~isempty(di(i,di(i,:)<0))
        error('not psd')
    end
    if isempty(calDim) || calDim == dim
        logdi = log(di(i,:));
        J = J + sum(logdi.*logdi,2) * Dt(i,i);
        dobji_dk = 2*logdi./di(i,:);
    else
        [ditemp, idxi] = sort(di(i,:), 2, 'descend');
        Vi = Vi(:,idxi);
        logdi = log(ditemp(1:calDim));
        penalty = max(log(ditemp(calDim+1:end)*epsilon_inv), 0);
        weight_penalty_sqrt = sqrt(weight_penalty);
        obji = [logdi, weight_penalty_sqrt * penalty];
        J = J + sum(obji.*obji, 2) * Dt(i,i);
        dobji_dk = 2*[logdi, weight_penalty * penalty]./ditemp;
%         [2*logdi./ditemp(1:calDim), ...
%             (penalty > 0) .* (2*penalty./ditemp(calDim+1:end))];
    end
    dHdzi = reshape(dHdz(:,:,i,:), [dim, dim, dim*N]);
    VTdHdzi = reshape(Vi'*dHdzi(:,:), [dim, dim, dim*N]);
    temp = reshape(permute(VTdHdzi, [1 3 2]), dim*dim*N, dim) * Vi;
    VTdHdziV = permute(reshape(temp, [dim, dim*N, dim]), [1 3 2]);
    %         temp = bsxfun(@times,Dt(i,i)*diag(dobji_dk), VTdHdziV);
    %         g = g + reshape(sum(sum(temp,1),2), dim*N, 1);
    VjTdHdziVj = reshape(VTdHdziV, [dim*dim, dim*N]);
    VjTdHdziVj = VjTdHdziVj(1:(dim+1):end, :);
    g = g + Dt(i,i)*(dobji_dk*VjTdHdziVj)';
end
Jmetric = J;
Jener = 0;
if (weight > 0 && ~isempty(Q))
    Jener = trace(Y'*Q*Y);
    J = J + weight * Jener;
    g = g + 2 * weight * reshape(Q*Y, length(g), 1);
end





