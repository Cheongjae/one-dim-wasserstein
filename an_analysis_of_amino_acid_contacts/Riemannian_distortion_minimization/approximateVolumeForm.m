function V = approximateVolumeForm(H, calDim)
% assume uniform distribution
if nargin < 3
    calDim = [];
end
N = size(H,3);
V = zeros(N,1);
if isempty(calDim)
    for i = 1:N
        V(i) = sqrt(det(H(:,:,i))^-1);
    end
else
    for i = 1:N
        d = eig(H(:,:,i));
        d = sort(d,'descend');
        V(i) = sqrt(det(diag(d(1:calDim)))^-1);
    end
end