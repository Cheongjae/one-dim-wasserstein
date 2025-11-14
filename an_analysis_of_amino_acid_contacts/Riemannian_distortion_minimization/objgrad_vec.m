function [J, g, H, dHdz, Jmetric, Jener] = objgrad_vec(x, L, Dt, Q, calDim, objtype)
if nargin < 6
    objtype = 'lamsq';
    if nargin < 5
        calDim = [];
        if nargin < 4
            Q = [];
        end
    end
end
if strcmp(objtype, 'lamsq')
    [J, g, H, dHdz, Jmetric, Jener] = objgrad_lamsq(reshape(x, size(L,1), length(x)/size(L,1)), L, Dt, Q, calDim);
elseif strcmp(objtype, 'Pndist')
    [J, g, H, dHdz, Jmetric, Jener] = objgrad_Pndist(reshape(x, size(L,1), length(x)/size(L,1)), L, Dt, Q, calDim);
elseif strcmp(objtype, 'RR')
    [J, g, H, dHdz, Jmetric, Jener] = objgrad_RR(reshape(x, size(L,1), length(x)/size(L,1)), L, Dt, Q, calDim);
elseif strcmp(objtype, 'condnum')
    [J, g, H, dHdz, Jmetric, Jener] = objgrad_condnum(reshape(x, size(L,1), length(x)/size(L,1)), L, Dt, Q, calDim);
elseif strcmp(objtype, 'unitvol')
    [J, g, H, dHdz, Jmetric, Jener] = objgrad_unitvol(reshape(x, size(L,1), length(x)/size(L,1)), L, Dt, Q, calDim);
end