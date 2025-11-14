function [J, g, H, dHdz, Jmetric, Jener] = objgrad_lamsq(Y, L, Dt, Q, calDim)
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
Id = eye(dim);
for i = 1:N
    %     if ~isempty(di(i,di(i,:)<0))
    %         error('not psd')
    %     end
    if isempty(calDim) || calDim == dim
        J = J + (0.5*trace(H(:,:,i)^2) - trace(H(:,:,i))) * Dt(i,i);
        dHdzi = reshape(dHdz(:,:,i,:), [dim, dim, dim*N]);
        temp2 = Dt(i,i)*(H(:,:,i) - Id);
        temp = bsxfun(@times, temp2, dHdzi);
        g = g + reshape(sum(sum(temp,1),2), dim*N, 1);
    else
        [Vi, Di] = eig(H(:,:,i));
        di(i,:) = diag(Di)';
        if ~isempty(di(i,di(i,:)<0))
            if length(di(i,di(i,:)<0)) > dim - calDim
                error('not psd')
            end
        end
        [ditemp, idxi] = sort(di(i,:), 2, 'descend');
        Vi = Vi(:,idxi);
        objdi = ditemp(1:calDim) - 1;
        weight_penalty_sqrt = sqrt(weight_penalty);
        penalty = max(ditemp(calDim+1:end) - epsilon, 0);
        obji = [objdi, weight_penalty_sqrt * penalty];
        J = J + sum(obji.*obji, 2) * Dt(i,i);
        
        dobji_dk = [2*objdi, weight_penalty * 2*penalty];
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
end
Jmetric = J;
Jener = 0;
if (weight > 0 && ~isempty(Q))
    Jener = trace(Y'*Q*Y);
    J = J + weight * Jener;
    g = g + 2 * weight * reshape(Q*Y, length(g), 1);
end




