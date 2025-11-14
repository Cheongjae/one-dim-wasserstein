function [J, g, H, dHdz, Jmetric, Jener] = objgrad_condnum(Y, L, Dt, Q, calDim)
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
    [Vi, Di] = eig(H(:,:,i));
    di(i,:) = diag(Di)';
%     temp2 = I(:,i)*L(i,:);
%     H2(:,:,i) = 0.5*Y'*(diag(L(i,:)) - temp2 - temp2')*Y;
    if ~isempty(di(i,di(i,:)<0))
        error('not psd')
    end
    [ditemp, idxi] = sort(di(i,:), 2, 'descend');
    if isempty(calDim) || calDim == dim
        idx1 = idxi(1);
        idx2 = idxi(dim);
        lam_min_inv = 1/ditemp(dim);
        J = J + ditemp(1)*lam_min_inv * Dt(i,i);        
        dHdzi = reshape(dHdz(:,:,i,:), [dim, dim, dim*N]);
        V1TdHdzi = reshape(Vi(:,idx1)'*dHdzi(:,:), [dim, dim*N]);
        V1TdHdziV1 = (Vi(:,idx1)'*V1TdHdzi)';
        V2TdHdzi = reshape(Vi(:,idx2)'*dHdzi(:,:), [dim, dim*N]);
        V2TdHdziV2 = (Vi(:,idx2)'*V2TdHdzi)';
        g = g + (V1TdHdziV1 - V2TdHdziV2*ditemp(1)*lam_min_inv) * lam_min_inv * Dt(i,i);
%         g2 = g2 + 2*(Di(idx,idx)-1)*reshape((diag(L(i,:))*Y - I(:,i)*L(i,:)*Y - L(i,:)'*Y(i,:))*Vi(:,idx)*Vi(:,idx)', dim*N,1) * Dt(i,i);
    else
        idx1 = idxi(1);
        idx2 = idxi(calDim);
        lam_min_inv = 1/ditemp(calDim);
        penalty = max(ditemp(calDim+1:end) - epsilon, 0);
        J = J + (ditemp(1)*lam_min_inv + weight_penalty * sum(penalty.*penalty, 2)) * Dt(i,i);        
        dHdzi = reshape(dHdz(:,:,i,:), [dim, dim, dim*N]);

        V1TdHdzi = reshape(Vi(:,idx1)'*dHdzi(:,:), [dim, dim*N]);
        V1TdHdziV1 = (Vi(:,idx1)'*V1TdHdzi)';
        V2TdHdzi = reshape(Vi(:,idx2)'*dHdzi(:,:), [dim, dim*N]);
        V2TdHdziV2 = (Vi(:,idx2)'*V2TdHdzi)';
        
        Vi_red = Vi(:,idxi(calDim+1:end));
        VTdHdzi = reshape(Vi_red'*dHdzi(:,:), [dim - calDim, dim, dim*N]);
        temp = reshape(permute(VTdHdzi, [1 3 2]), (dim - calDim)*dim*N, dim) * Vi_red;
        VTdHdziV = permute(reshape(temp, [dim - calDim, dim*N, dim - calDim]), [1 3 2]);
        VjTdHdziVj = reshape(VTdHdziV, [(dim - calDim)*(dim - calDim), dim*N]);
        VjTdHdziVj = VjTdHdziVj(1:((dim - calDim)+1):end, :);
        
        g = g + ((V1TdHdziV1 - V2TdHdziV2*ditemp(1)*lam_min_inv) * lam_min_inv ...
            + weight_penalty * 2*(penalty * VjTdHdziVj)') * Dt(i,i);
    end
end
Jmetric = J;
Jener = 0;
if (weight > 0 && ~isempty(Q))
    Jener = trace(Y'*Q*Y);
    J = J + weight * Jener;
    g = g + 2 * weight * reshape(Q*Y, length(g), 1);
end




