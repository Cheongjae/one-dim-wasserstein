function [Y_new, J_af, H_new, optTime, Jval, exitFlag, output] = distortion_minimization(Y_ic, L_dm, Dt, objftn, options)
% objftn: one of 'lamsq', 'Pndist', 'RR', 'unitvol', 'condnum'
% L_dm: normalized graph Laplacian
% Dt: d_tilde in diffusion map

% initial guess
x0 = reshape(Y_ic, size(Y_ic,1)*size(Y_ic,2), 1);

% run optimization
tic
[xopt,Jval,exitFlag,output] = fminunc(@(x)  objgrad_vec(x, L_dm, Dt, [], [], objftn), x0, options);
optTime = toc;

Y_new = reshape(xopt, size(Y_ic,1), size(Y_ic,2));
[J_af, H_new] = obj(Y_new, L_dm, Dt, [], [], objftn);