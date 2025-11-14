function [J, g, H, dHdz, Jmetric, Jener] = objgrad_unitvol(Y, L, Dt, Q, calDim)
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
% g2 = zeros(size(Y,1)*size(Y,2), 1);
N = size(L,1);
dim = size(Y,2);

% calculate components of metric inv and deriv of metric inv
[H, dHdz] = estimateMetricInv(Y, L);
di = zeros(N,dim);

% calculate obj ftn and grad
J = 0;
for i = 1:N
    %     if ~isempty(di(i,di(i,:)<0))
    %         error('not psd')
    %     end
    if isempty(calDim) || calDim == dim
        detH = det(H(:,:,i));
        logdetH = log(detH);
        J = J + logdetH^2 * Dt(i,i);
        dHdzi = reshape(dHdz(:,:,i,:), [dim, dim, dim*N]);
        HinvT = inv(H(:,:,i))';
        temp = bsxfun(@times, HinvT, dHdzi);
        g = g + 2 * reshape(sum(sum(temp,1),2), dim*N, 1) * Dt(i,i) * logdetH;
    else
        [Vi, Di] = eig(H(:,:,i));
        di(i,:) = diag(Di)';
        if ~isempty(di(i,di(i,:)<0))
            error('not psd')
        end
        [ditemp, idxi] = sort(di(i,:), 2, 'descend');
        penalty = max(ditemp(calDim+1:end) - epsilon, 0);
        logdetH_calDim = sum(log(ditemp(1:calDim)), 2);
        J = J + (logdetH_calDim^2 + weight_penalty * sum(penalty.*penalty, 2)) * Dt(i,i);
        dHdzi = reshape(dHdz(:,:,i,:), [dim, dim, dim*N]);
        
        dobji_dk = [2*logdetH_calDim./ditemp(1:calDim), weight_penalty * 2*penalty];
        
        Vi = Vi(:,idxi);
        VTdHdzi = reshape(Vi'*dHdzi(:,:), [dim, dim, dim*N]);
        temp = reshape(permute(VTdHdzi, [1 3 2]), dim*dim*N, dim) * Vi;
        VTdHdziV = permute(reshape(temp, [dim, dim*N, dim]), [1 3 2]);
        VjTdHdziVj = reshape(VTdHdziV, [dim*dim, dim*N]);
        VjTdHdziVj = VjTdHdziVj(1:(dim+1):end, :);
        g = g + Dt(i,i)*(dobji_dk*VjTdHdziVj)';
    end
end
Jmetric = J;
Jener = 0;
if (weight > 0 && ~isempty(Q))
    Jener = trace(Y'*Q*Y);
    J = J + weight * Jener;
    g = g + 2 * weight * reshape(Q*Y, length(g), 1);
end




