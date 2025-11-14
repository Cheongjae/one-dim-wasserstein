function [J, g, H, dHdz, Jmetric, Jener] = objgrad_RR(Y, L, Dt, Q, calDim)
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
g = zeros(size(Y,1)*size(Y,2), 1);
% g2 = zeros(size(Y,1)*size(Y,2), 1);
N = size(L,1);
dim = size(Y,2);

if ~isempty(calDim)
    if calDim > dim
        error('calDim should be less than dim')
    end
end
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
    if isempty(calDim) || calDim == dim
        temp = di(i,:) - ones(1,dim);
        [disqtemp, idxi] = sort(temp.*temp, 2, 'descend');
        idx = idxi(1);
        J = J + disqtemp(1) * Dt(i,i);        
        dHdzi = reshape(dHdz(:,:,i,:), [dim, dim, dim*N]);
        VTdHdzi = reshape(Vi(:,idx)'*dHdzi(:,:), [dim, dim*N]);
        VTdHdziV = (Vi(:,idx)'*VTdHdzi)';
        g = g + 2*(Di(idx,idx)-1)*VTdHdziV * Dt(i,i);
%         g2 = g2 + 2*(Di(idx,idx)-1)*reshape((diag(L(i,:))*Y - I(:,i)*L(i,:)*Y - L(i,:)'*Y(i,:))*Vi(:,idx)*Vi(:,idx)', dim*N,1) * Dt(i,i);
    else
        [ditemp, idxi] = sort(di(i,:), 2, 'descend');
        temp = [ditemp(1:calDim) - 1, ditemp(calDim+1:end) * epsilon_inv];
        [disqtemp, idxi_2] = sort(temp.*temp, 2, 'descend');
        idx = idxi(idxi_2(1));
        J = J + disqtemp(1) * Dt(i,i);        
        dHdzi = reshape(dHdz(:,:,i,:), [dim, dim, dim*N]);
        VTdHdzi = reshape(Vi(:,idx)'*dHdzi(:,:), [dim, dim*N]);
        VTdHdziV = (Vi(:,idx)'*VTdHdzi)';
        if idxi_2(1) < calDim + 1
            g = g + 2*(Di(idx,idx)-1)*VTdHdziV * Dt(i,i);
        else
            g = g + 2*(Di(idx,idx) * epsilon_inv^2)*VTdHdziV * Dt(i,i);
        end
    end
end
Jmetric = J;
Jener = 0;
if (weight > 0 && ~isempty(Q))
    Jener = trace(Y'*Q*Y);
    J = J + weight * Jener;
    g = g + 2 * weight * reshape(Q*Y, length(g), 1);
end




