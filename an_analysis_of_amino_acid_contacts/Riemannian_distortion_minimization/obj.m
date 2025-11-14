function [J, H, Jmetric, Jener, di, sig_vec] = obj(Y, L, Dt, Q, calDim, objtype)
% Y: data (num of data X dim)
% L: Laplace-Beltrami operator
% Dt: normalized diagonal matrix
% Q: quadratic form to calculate energy
% calDim: dimension for calcuate distortion (may be different from dim, and corresponds to intrinsic dim in RR paper)
if nargin < 6
    objtype = 'lamsq';
if nargin < 5
    calDim = [];
    if nargin < 4
       Q = []; 
    end
end
end
di = [];
if strcmp(objtype, 'lamsq')
    [J, H, Jmetric, Jener, sig_vec] = obj_lamsq(Y, L, Dt, Q, calDim);
elseif strcmp(objtype, 'Pndist')
    [J, H, di, Jmetric, Jener, sig_vec] = obj_Pndist(Y, L, Dt, Q, calDim);
elseif strcmp(objtype, 'RR')
    [J, H, Jmetric, Jener, sig_vec] = obj_RR(Y, L, Dt, Q, calDim);
elseif strcmp(objtype, 'condnum')
    [J, H, Jmetric, Jener, sig_vec] = obj_condnum(Y, L, Dt, Q, calDim);
elseif strcmp(objtype, 'unitvol')
    [J, H, Jmetric, Jener, sig_vec] = obj_unitvol(Y, L, Dt, Q, calDim);
end