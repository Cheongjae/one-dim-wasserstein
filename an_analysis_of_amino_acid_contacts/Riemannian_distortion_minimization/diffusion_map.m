function [Y, Lap, Dt, D, L, dL, Q, vL] = diffusion_map(X, h, dim, distMat, alpha, deg, eigdim)
% X: data (numData X dimX)
% h: kernel bandwidth parameter
% dim: dimension of new embedding (dimY)
% alpha: normalization constant (alpha = 0: Laplacian eigenmap, alpha = 1: diffusion map)
% deg: the number of interations for diffusion process
if nargin < 7
    eigdim = dim;
    if nargin < 6
        deg = [];
        if nargin < 5
            alpha = [];
            if nargin < 4
                distMat = [];
            end
        end
    end
end
if isempty(alpha)
    alpha = 1;
end
if isempty(deg)
    deg = 1;
end
if isempty(X) && ~isempty(distMat)
    N = size(distMat, 1);
    W = exp(distMat/h);
elseif isempty(X) && isempty(distMat)
    error('data or distMat should be given!!!')
else
    N = size(X,1);
    if dim > size(X,2)
        error('dim should be less than dimX!!!')
    end
    if size(distMat, 1) ~= N
        distMat = [];
    end
    if isempty(distMat)
        temp = zeros(N,N);
        for i = 1:N
            for j = 1:i
                temp(i,j) = -sum((X(i,:) - X(j,:)).*(X(i,:) - X(j,:)), 2);
            end
        end
        temp = temp + temp' - diag(diag(temp));
        W = exp(temp/h);
    else
        W = exp(distMat/h);
    end
end
%%%%%%%%%%%%%%%%%%
% for i = 1:N
%     W(i,i) = 0;
% end
%%%%%%%%%%%%%%%%%%
D = diag((W*ones(N,1)));
Dinv = diag(1./(W*ones(N,1)));
Wt = Dinv*W*Dinv;
Dt = diag(Wt*ones(N,1));
Q = Dt - Wt;
L = Dt^-1*Wt;
if alpha ~= 1
    for i = 1:N
        W(i,i) = 0;
    end
    Dinv = diag(1./(W*ones(N,1)));
    L_le = Dinv*W;
    [vL, dL] = eigs(L_le, eigdim+1);
else
    [vL, dL] = eigs(L, eigdim+1);
end

dL = diag(diag(dL(2:end,2:end)).^deg);
vL = vL(:,2:end);

if alpha == 1
    % diffusion map
    Y = vL*dL(:,1:dim);
else
    % laplacian eigenmap
    Y = vL(:,1:dim);
end
Lap = (L - eye(N)) / (0.25 * h);  % from Perrault-Joncas paper