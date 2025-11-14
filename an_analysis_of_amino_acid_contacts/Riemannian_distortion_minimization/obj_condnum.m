function [J, H, Jmetric, Jener, sig_vec] = obj_condnum(Y, L, Dt, Q, calDim)
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
    [~, Di] = eig(H(:,:,i));
    di(i,:) = diag(Di)';
%     temp2 = I(:,i)*L(i,:);
%     H2(:,:,i) = 0.5*Y'*(diag(L(i,:)) - temp2 - temp2')*Y;
    if ~isempty(di(i,di(i,:)<0))
        error('not psd')
    end
    [ditemp] = sort(di(i,:), 2, 'descend');
    if isempty(calDim) || calDim == dim
        J = J + ditemp(1)/ditemp(dim) * Dt(i,i);
        sig_vec(i) = ditemp(1)/ditemp(dim);
%         g2 = g2 + 2*(Di(idx,idx)-1)*reshape((diag(L(i,:))*Y - I(:,i)*L(i,:)*Y - L(i,:)'*Y(i,:))*Vi(:,idx)*Vi(:,idx)', dim*N,1) * Dt(i,i);
    else
        lam_min_inv = 1/ditemp(calDim);
        penalty = max(ditemp(calDim+1:end) - epsilon, 0);
        J = J + (ditemp(1)*lam_min_inv + weight_penalty * sum(penalty.*penalty, 2)) * Dt(i,i);
        sig_vec(i) = ditemp(1)*lam_min_inv;
        %%%%%%%%%%%%%%%%%%% should be modified
%         [ditemp, idxi] = sort(di(i,:), 2, 'descend');
%         ditemp = ditemp(1:calDim);
%         logdi = log(ditemp);
%         J = J + sum(logdi.*logdi,2) * Dt(i,i);
%         two_logdi_over_di = 2*logdi./ditemp;
%         for j = 1:dim_bc
%             temp = diag(Vi'*dHdz(:,:,i,j)*Vi);
%             g(j) = g(j) + sum(two_logdi_over_di .* temp(idxi(1:calDim))', 2) * Dt(i,i);
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             %         dHdz_a(:,:,i,j) = dHdz_a(:,:,i,j) + dHdz_a(:,:,i,j)' - diag(diag(dHdz_a(:,:,i,j)));
%             %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         end
    end
end
Jmetric = J;
Jener = 0;
if (weight > 0 && ~isempty(Q))
    Jener = trace(Y'*Q*Y);
    J = J + weight * Jener;
end




