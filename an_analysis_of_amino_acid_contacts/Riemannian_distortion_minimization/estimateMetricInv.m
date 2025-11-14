function [H, dHdz] = estimateMetricInv(Y, L, calculateGrad, dYdz)
% Y: data (num of data X dim)
% L: Laplace-Beltrami operator
% calculateGrad: option to calculate gradient
% dYdz: derivative of Y with respect to optimization variable z
if nargin < 4
    dYdz = [];
    if nargin < 3
        calculateGrad = true;
    end
end
% calculate components of metric inv and deriv of metric inv
N = size(L,1);
dim = size(Y,2);
I = eye(N);
H = zeros(dim,dim,N);
if calculateGrad
    if isempty(dYdz)
        Nvar = N;
    else
        Nvar = size(dYdz, 2);
    end
    dHdz = zeros(dim,dim,N,dim*Nvar);
else
    dHdz = [];
end
n = dim*(dim+1)/2;
rowNum = zeros(1,n);
colNum = zeros(1,n);
k = 0;
for i = 1:dim
    for j = 1:i
        k = k+1;
        rowNum(k) = i;
        colNum(k) = j;
    end
end
LY = L*Y;

for k = 1:n
    r = rowNum(k);
    c = colNum(k);
    H(r,c,:) = 0.5*(L*(Y(:,r).*Y(:,c)) - Y(:,r).*(LY(:,c)) - Y(:,c).*(LY(:,r)));
    if r ~= c
        H(c,r,:) = H(r,c,:);
    end
    if calculateGrad
        if isempty(dYdz)
            %%%%%%%%%%%%%%%%%%% ver1
            for l = 1:Nvar
                dHdz(r,c,:,(r-1)*Nvar+l) = dHdz(r,c,:,(r-1)*Nvar+l) ...
                    + reshape(0.5*(L(:,l)*Y(l,c) - I(:,l)*LY(l,c) - Y(:,c).*(L(:,l))), [1,1,N]);
                dHdz(r,c,:,(c-1)*Nvar+l) = dHdz(r,c,:,(c-1)*Nvar+l) ...
                    + reshape(0.5*(L(:,l)*Y(l,r) - Y(:,r).*(L(:,l)) - I(:,l)*LY(l,r)), [1,1,N]);
            end
        else
            %%%%%%%%%%%%%%%%%%% ver2
            temp = zeros(N, dim*N);
            for l = 1:N
                temp(:, (r-1)*N+l) = temp(:, (r-1)*N+l) ...
                    + 0.5*(L(:,l)*Y(l,c) - I(:,l)*LY(l,c) - Y(:,c).*(L(:,l)));
                temp(:, (c-1)*N+l) = temp(:, (c-1)*N+l) ...
                    + 0.5*(L(:,l)*Y(l,r) - Y(:,r).*(L(:,l)) - I(:,l)*LY(l,r));
            end
            dHdz(r,c,:,:) = reshape(permute(reshape(...
                reshape(permute(reshape(temp, [N, N, dim]), [1,3,2]), [N*dim, N]) * dYdz, ...
                [N, dim, Nvar]), [1,3,2]),[1,1,N,Nvar*dim]);
        end
        if r ~= c
            dHdz(c,r,:,:) = dHdz(r,c,:,:);
        end
    end
end
