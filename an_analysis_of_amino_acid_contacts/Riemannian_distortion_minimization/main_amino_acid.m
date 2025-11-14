close all
clear variables
global weight
weight = 0;

%% Amino acid contact frequency data
% load similarity graph
load('wass_scaled_pdist.mat')

distMat_Was_init = -data;

load('KL_scaled_pdist.mat')

distMat_KL_init = -data;

offset = 0.1;
distMat_Was = distMat_Was_init - offset;
distMat_KL = distMat_KL_init - offset;
for i = 1:20
    distMat_Was(i,i) = 0;
    distMat_KL(i,i) = 0;
end


figure()
for i = 1:20
    plot(sort(-distMat_Was(i,:)))
    hold on
end
ylim([0, 1 + offset])

figure()
for i = 1:20
    plot(sort(-distMat_KL(i,:)))
    hold on
end
ylim([0, 1 + offset])

N = 20;


% other settings
dim = 2;

markers = ['o', '+', '*', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h', ...
    'o', '+', '*', 'x', 's', 'd', '^', 'v', '>', '<', 'p', 'h'];
legends = {'ALA',  'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', ...
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'};



%% Optimize distortion objective 
close all

% references for hydrophobicity
rose_table = [0.74, 0.64, 0.63, 0.62, 0.91, 0.62, 0.62, 0.72, 0.78, 0.88, 0.85, 0.52, 0.85, 0.88, 0.64, 0.66, 0.70, 0.85, 0.76, 0.86];

engelman_table = [1.60, -12.3, -4.80, -9.20, 2.00, -8.20, -4.10, 1.00, -3.00, 3.10, 2.80, -8.80, 3.40, 3.70, -0.20, 0.60, 1.20, 1.90, -0.70, 2.60];


% select objective function
objftn = 'lamsq';
% objftn = 'Pndist';
% objftn = 'RR';
% objftn = 'unitvol';
% objftn = 'condnum';

% set optimization option
hessOption = 'bfgs';       % 'bfgs', 'dfp', 'steepdesc'
options = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'GradObj','on', ...
    'MaxIter', 10000, 'Display', 'iter', 'TolFun', 1e-5, 'HessUpdate', hessOption);

% hyperparameters
% offsets = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
% hs = [0.1, 0.2, 0.4, 0.8];
offsets = [0.4];
hs = [0.2];

% result variables
norm_set_Was = zeros(20, length(offsets), length(hs));
norm_set_KL = zeros(20, length(offsets), length(hs));

kendall_coef_Was = zeros(length(offsets), length(hs));
kendall_coef_KL = zeros(length(offsets), length(hs));
kendall_coef_top_Was = zeros(length(offsets), length(hs));
kendall_coef_top_KL = zeros(length(offsets), length(hs));

distort_dist_bf_Was = zeros(length(offsets));
distort_dist_bf_KL = zeros(length(offsets));
distort_dist_af_Was = zeros(length(offsets), length(hs));
distort_dist_af_KL = zeros(length(offsets), length(hs));

local_distort_dist_bf_Was = zeros(length(offsets));
local_distort_dist_bf_KL = zeros(length(offsets));
local_distort_dist_af_Was = zeros(length(offsets), length(hs));
local_distort_dist_af_KL = zeros(length(offsets), length(hs));

top_lists = [0,4,7,8,9,10,12,13,16,17,18,19] + 1;
% top_lists = [0,4,7,8,9,10,12,13,17,18,19] + 1;

k_local = 5;
[local_dist_Was, idx_Was] = sort(-distMat_Was, 2);
idx_Was = idx_Was(:,2:k_local+1);
local_dist_Was = local_dist_Was(:,2:k_local+1);
[local_dist_KL, idx_KL] = sort(-distMat_KL, 2);
idx_KL = idx_KL(:,2:k_local+1);
local_dist_KL = local_dist_KL(:,2:k_local+1);
for j = 1:length(offsets)
    offset = offsets(j);
    distMat_Was = distMat_Was_init - offset;
    distMat_KL = distMat_KL_init - offset;
    for i = 1:20
        distMat_Was(i,i) = 0;
        distMat_KL(i,i) = 0;
    end

    % isomap initial guess for Wasserstein embedding
    k = 5;
    dim2 = 10;
    [Y_iso1, G_iso, V_iso, D_iso, T_iso, S_iso] = isomap([], k, dim2, [], -distMat_Was);
    % isomap initial guess for KL-divergence embedding
    k = 5;
    dim2 = 10;
    [Y_iso2, G_iso, V_iso, D_iso, T_iso, S_iso] = isomap([], k, dim2, [], -distMat_KL);
    
    pairDistAll1_bf = getPairwiseDist(Y_iso1);
    pairDistAll2_bf = getPairwiseDist(Y_iso2);
    distort_dist_bf_Was(j) = norm(pairDistAll1_bf + distMat_Was,'fro')^2/N/(N-1)/2; 
    distort_dist_bf_KL(j) = norm(pairDistAll2_bf + distMat_KL,'fro')^2/N/(N-1)/2; 

    
    local_pairDistAll1_bf = getPairwiseDist(Y_iso1, k_local);
    local_pairDistAll2_bf = getPairwiseDist(Y_iso2, k_local);
    local_distort_dist_bf_Was(j) = norm(local_pairDistAll1_bf + local_dist_Was,'fro')^2/k_local/(k_local-1)/2; 
    local_distort_dist_bf_KL(j) = norm(local_pairDistAll2_bf + local_dist_KL,'fro')^2/k_local/(k_local-1)/2; 

    figure
    subplot(1,2,1)
    drawAminoAcids(Y_iso1, markers, legends, ['Wass iso init (offset: ', num2str(offset), ')'], false)
    subplot(1,2,2)
    drawAminoAcids(Y_iso2, markers, legends, ['KL iso init (offset: ', num2str(offset), ')'], false)
    set(gcf,'Position',[100 100 1400 800])
%     saveas(gcf,['./figs_sq/iso init offset_', num2str(offset), '.png'])


    % perform optimization
    for i=1:length(hs)
        h = hs(i);

        % nearly isometric embedding for Wasserstein
        [Y_dm1, L_dm1, Dt1, ~, ~, ~, ~] = diffusion_map([], h, dim, -distMat_Was.*distMat_Was);
        [Y_new1, J_af, H_new, optTime, Jval, exitFlag, output] ...
            = distortion_minimization(Y_iso1(:,1:2), L_dm1, Dt1, objftn, options);
    
        
        % nearly isometric embedding for KL-divergence
        [Y_dm2, L_dm2, Dt2, ~, ~, ~, ~] = diffusion_map([], h, dim, -distMat_KL.*distMat_KL);
        [Y_new2, J_af, H_new, optTime, Jval, exitFlag, output] ...
            = distortion_minimization(Y_iso2(:,1:2), L_dm2, Dt2, objftn, options);

        norm_set_Was(:,j,i) = sum(Y_new1.*Y_new1, 2);
        norm_set_KL(:,j,i) = sum(Y_new2.*Y_new2, 2);

        figure
        subplot(1,2,1)
        drawAminoAcids(Y_new1, markers, legends, ...
            ['Wass embedding (offset: ', num2str(offset), ', h: ', num2str(h),')'], false)
        subplot(1,2,2)
        drawAminoAcids(Y_new2, markers, legends, ...
            ['KL embedding (offset: ', num2str(offset), ', h: ', num2str(h), ')'], false)
        set(gcf,'Position',[100 100 1400 800])
%         saveas(gcf,['./figs_sq/embeddings offset_',num2str(offset),'_h_',num2str(h),'.png'])
    
        figure
        X = categorical(legends);
        b = bar(X, norm_set_Was(:,j,i), 'b');
        b.FaceAlpha = 0.2;
        hold on
        b = bar(X, norm_set_KL(:,j,i), 'r');
        b.FaceAlpha = 0.2;
        set(gcf,'Position',[100 100 1400 800])
%         saveas(gcf,['./figs_sq/norms offset_',num2str(offset),'_h_',num2str(h),'.png'])

        [kendall_coef_Was(j,i), ~] = corr(norm_set_Was(:,j,i),rose_table','type','Kendall');
        [kendall_coef_KL(j,i), ~] = corr(norm_set_KL(:,j,i),rose_table','type','Kendall');
        [kendall_coef_top_Was(j,i), ~] = corr(norm_set_Was(top_lists,j,i),rose_table(top_lists)','type','Kendall');
        [kendall_coef_top_KL(j,i), ~] = corr(norm_set_KL(top_lists,j,i),rose_table(top_lists)','type','Kendall');

        
        pairDistAll1_af = getPairwiseDist(Y_new1);
        pairDistAll2_af = getPairwiseDist(Y_new2);
        distort_dist_af_Was(j,i) = norm(pairDistAll1_af + distMat_Was,'fro')^2/N/(N-1)/2;
        distort_dist_af_KL(j,i) = norm(pairDistAll2_af + distMat_KL,'fro')^2/N/(N-1)/2;
        
        local_pairDistAll1_af = getPairwiseDist(Y_new1, k_local);
        local_pairDistAll2_af = getPairwiseDist(Y_new2, k_local);
        local_distort_dist_af_Was(j,i) = norm(local_pairDistAll1_af + local_dist_Was,'fro')^2/k_local/(k_local-1)/2;
        local_distort_dist_af_KL(j,i) = norm(local_pairDistAll2_af + local_dist_KL,'fro')^2/k_local/(k_local-1)/2;

    end
end

%% Kendall tau coefficients
% reference: https://resources.qiagenbioinformatics.com/manuals/clcgenomicsworkbench/650/Hydrophobicity_scales.html
KD_table = [1.80, -4.50, -3.50, -3.50, 2.50, -3.50, -3.50, -0.40, -3.20, 4.50, 3.80, -3.90, 1.90, 2.80, -1.60, -0.80, -0.70, -0.90, -1.30, 4.20];
HW_table = [-0.50, 3.00, 0.20, 3.00, -1.00, 3.00, 0.20, 0.00, -0.50, -1.80, -1.80, 3.00, -1.30, -2.50, 0.00, 0.30, -0.40, -3.40, -2.30, -1.50];
Cornette_table = [0.20, 1.40, -0.50, -3.10, 4.10, -1.80, -2.80, 0.00, 0.50, 4.80, 5.70, -3.10, 4.20, 4.40, -2.20, -0.50, -1.90, 1.00, 3.20, 4.70];
Eisenberg_table = [0.62, -2.53, -0.78, -0.90, 0.29, -0.74, -0.85, 0.48, -0.40, 1.38, 1.06, -1.50, 0.64, 1.19, 0.12, -0.18, -0.05, 0.81, 0.26, 1.08];
Rose_table = [0.74, 0.64, 0.63, 0.62, 0.91, 0.62, 0.62, 0.72, 0.78, 0.88, 0.85, 0.52, 0.85, 0.88, 0.64, 0.66, 0.70, 0.85, 0.76, 0.86];
Janin_table = [0.30, -1.40, -0.50, -0.60, 0.90, -0.70, -0.70, 0.30, -0.10, 0.70, 0.50, -1.80, 0.40, 0.50, -0.30, -0.10, -0.20, 0.30, -0.40, 0.60];
EGES_table = [1.60, -12.3, -4.80, -9.20, 2.00, -8.20, -4.10, 1.00, -3.00, 3.10, 2.80, -8.80, 3.40, 3.70, -0.20, 0.60, 1.20, 1.90, -0.70, 2.60];

hydrophobicity_table = [KD_table; HW_table; Cornette_table; Eisenberg_table; 
    Rose_table; Janin_table; EGES_table];
num_top = 12;
kendall_coef_all_Was = zeros(size(hydrophobicity_table,1), length(offsets), length(hs));
kendall_coef_all_KL = zeros(size(hydrophobicity_table,1), length(offsets), length(hs));
kendall_coef_top_all_Was = zeros(size(hydrophobicity_table,1), length(offsets), length(hs));
kendall_coef_top_all_KL = zeros(size(hydrophobicity_table,1), length(offsets), length(hs));
for k = 1:size(hydrophobicity_table,1)
    [~, top_k_lists] = sort(hydrophobicity_table(k,:), 'descend');
    top_k_lists = sort(top_k_lists(1:num_top));
    for j = 1:length(offsets)
        for i=1:length(hs)
            [kendall_coef_all_Was(k,j,i), ~] = corr(norm_set_Was(:,j,i),hydrophobicity_table(k,:)','type','Kendall');
            [kendall_coef_all_KL(k,j,i), ~] = corr(norm_set_KL(:,j,i),hydrophobicity_table(k,:)','type','Kendall');
            [kendall_coef_top_all_Was(k,j,i), ~] = corr(norm_set_Was(top_k_lists,j,i),hydrophobicity_table(k,top_k_lists)','type','Kendall');
            [kendall_coef_top_all_KL(k,j,i), ~] = corr(norm_set_KL(top_k_lists,j,i),hydrophobicity_table(k,top_k_lists)','type','Kendall');
        end
    end
%     display([k, sum(kendall_coef_all_Was(k,:,:) > kendall_coef_all_KL(k,:,:), [1,2,3]), sum(kendall_coef_all_Was(k,:,:) < kendall_coef_all_KL(k,:,:), [1,2,3])])
%     display([k, sum(kendall_coef_top_all_Was(k,:,:) > kendall_coef_top_all_KL(k,:,:), [1,2,3]), sum(kendall_coef_top_all_Was(k,:,:) < kendall_coef_top_all_KL(k,:,:), [1,2,3])])
end

