function [J, H, di] = obj_distortion(L, Y, Dt, calDim)
% L: Laplace-Beltrami operator
% Y: data (number of data X dim)
% Dt: normalized diagonal matrix
% calDim: dimension for calcuate distortion (may be different from dim)
if nargin < 4
    calDim = [];
end
dim = size(Y,2);
N = size(Y,1);
H = zeros(dim,dim,N);
di = zeros(N,dim);
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
for k = 1:n
     r = rowNum(k);
     c = colNum(k);
     H(r,c,:) = 0.5*(L*(Y(:,r).*Y(:,c)) - Y(:,r).*(L*Y(:,c)) - Y(:,c).*(L*Y(:,r)));
     if r ~= c
         H(c,r,:) = H(r,c,:);
     end
end
J = 0;
for i = 1:N
    di(i,:) = eig(H(:,:,i))';
    if isempty(calDim)
        logdi = log(di(i,:));
    else
        if ~isempty(di(i,di(i,:)<0))
            if length(di(i,di(i,:)<0)) > dim - calDim
                error('not psd')
            end
        end
        di_temp = sort(di(i,:), 'descend');
        logdi = log(di_temp(1:calDim));
    end
    J = J + sum(logdi.*logdi,2) * Dt(i,i);
end
a = 1;