function [data, color, s] = generateNonUniformSwissRoll(th0, thf, z0, zf, N, ratio, holeRange, pdftype)
% holeRange: [xleft, xright; ybottom, ytop]
% pdftype 1: radial symmetric, sparse center
% pdftype 2: oscillating density along s direction, uniform in z direction
% pdftype 3: dense center

if nargin < 8
    pdftype = 1;
    if nargin < 7
        holeRange = [];
        if nargin < 6
            ratio = 0.5;
        end
    end
end
s0 = swissRoll_th2s(th0);
sf = swissRoll_th2s(thf);
if isempty(holeRange)
    if pdftype == 1
        R = 0.5;
        rand_r1 = rand(1, N - fix(N*ratio));
        rand_r2 = rand(1, fix(N*ratio));
        r = [invQumulFunc_r1(rand_r1, R), invQumulFunc_r2(rand_r2, R)];
        phi = 2*pi*rand(1,N);
        smean = (sf + s0)/2;
        zmean = (zf + z0)/2;
        s = smean + (sf-s0)*r.*cos(phi);
        z = zmean + (zf-z0)*r.*sin(phi);
%         s = invQumulFunc(rand_s, s0, sf, ratio);
%         z = invQumulFunc(rand_z, z0, zf, ratio);
%         z = z0 + (zf - z0)*rand_z;
    elseif pdftype == 2
        num_bin = 7;
        size_bin = 1/num_bin;
        Nh = fix(num_bin/2);
        n = fix(N / ((1+ratio)*Nh + 1));
        rand_s = [];
        for i = 1:num_bin
            if i < num_bin
               if mod(i,2) == 1
                   temp = rand(1,n)*size_bin + (i-1)*size_bin;
               else
                   temp = rand(1,fix(ratio*n))*size_bin + (i-1)*size_bin;
               end
            else
                temp = rand(1,N - length(rand_s))*size_bin + (i-1)*size_bin;
            end
            rand_s = [rand_s, temp];
        end
        s = s0 + (sf - s0)*rand_s;
        rand_z = rand(1,N);
        z = z0 + (zf - z0)*rand_z;
    else
        rand_s = rand(1,N);
        rand_z = rand(1,N);
        s = invQumulFunc2(rand_s, s0, sf, ratio);
        z = invQumulFunc2(rand_z, z0, zf, ratio);
    end
else
    Ntemp = fix(2*N/(1-(holeRange(1,2) - holeRange(1,1))*(holeRange(2,2) - holeRange(2,1))));
    rand_s = rand(1,Ntemp);
    rand_z = rand(1,Ntemp);
    if pdftype == 1
        R = 0.5;
        rand_r1 = rand(1, Ntemp - fix(Ntemp*ratio));
        rand_r2 = rand(1, fix(Ntemp*ratio));
        r = [invQumulFunc_r1(rand_r1, R), invQumulFunc_r2(rand_r2, R)];
        phi = 2*pi*rand(1,Ntemp);
        smean = (sf + s0)/2;
        zmean = (zf + z0)/2;
        s = smean + (sf-s0)*r.*cos(phi);
        z = zmean + (zf-z0)*r.*sin(phi);
%         s = invQumulFunc(rand_s, s0, sf, ratio);
%         z = invQumulFunc(rand_z, z0, zf, ratio);
%         z = z0 + (zf - z0)*rand_z;
    elseif pdftype == 2
        num_bin = 5;
        size_bin = 1/num_bin;
        Nh = fix(num_bin/2);
        n = fix(Ntemp / ((1+ratio)*Nh + 1));
        rand_s = [];
        for i = 1:num_bin
            if i < num_bin
               if mod(i,2) == 1
                   temp = rand(1,n)*size_bin + (i-1)*size_bin;
               else
                   temp = rand(1,fix(ratio*n))*size_bin + (i-1)*size_bin;
               end
            else
                temp = rand(1,Ntemp - length(rand_s))*size_bin + (i-1)*size_bin;
            end
            rand_s = [rand_s, temp];
        end
        s = s0 + (sf - s0)*rand_s;
        rand_z = rand(1,Ntemp);
        z = z0 + (zf - z0)*rand_z;
    else
        s = invQumulFunc2(rand_s, s0, sf, ratio);
        z = invQumulFunc2(rand_z, z0, zf, ratio);
    end
    temp = and(and(s < s0 + (sf - s0)*holeRange(1,2), s > s0 + (sf - s0)*holeRange(1,1)), ...
        and(z < z0 + (zf - z0)*holeRange(2,2), z > z0 + (zf - z0)*holeRange(2,1)));
    s = s(~temp);
    z = z(~temp);
    if length(s) < N
        N = length(s);
    end
    idxset = randperm(length(s),N);
    s = s(idxset);
    z = z(idxset);
end
color = (s-s0)/(sf-s0) - 0.5;
th = zeros(1,N);
for i=1:N
    th(i) = swissRoll_s2th(s(i));
end
data = [th.*cos(th); th.*sin(th); z];
data = data';
color = color';
end
%% considered inverse qumulative functions
function s = invQumulFunc(q, s0, sf, ratio)
% linear pdf, sparse center
q(q<0) = 0;
q(q>1) = 1;
alpha = 2/(sf-s0)/(1+ratio);
a1 = (ratio-1)*alpha/(sf - s0);
a2 = -a1;
b = alpha;
c1 = -q(q<0.5);
c2 = 1-q(q>=0.5);
s = zeros(1,length(q));
s(q<0.5) = s0 + (-b + sqrt(b^2 - 4*a1*c1))/2/a1;
s(q>=0.5) = sf + (-b + sqrt(b^2 - 4*a2*c2))/2/a2;
% figure()
% scatter(s,q);
end
function s = invQumulFunc2(q, s0, sf, ratio)
% linear pdf, dense center
q(q<0) = 0;
q(q>1) = 1;
alpha = 2/(sf-s0)/(1+ratio);
a1 = (1-ratio)*alpha/(sf - s0);
a2 = -a1;
b = ratio*alpha;
c1 = -q(q<0.5);
c2 = 1-q(q>=0.5);
s = zeros(1,length(q));
s(q<0.5) = s0 + (-b + sqrt(b^2 - 4*a1*c1))/2/a1;
s(q>=0.5) = sf + (-b + sqrt(b^2 - 4*a2*c2))/2/a2;
% figure()
% scatter(s,q);
end
function r = invQumulFunc_r1(q, R)
% linear radial pdf, sparse center
q(q<0) = 0;
q(q>1) = 1;
alpha = 3/R^3;
r = (3*q/alpha).^(1/3);
end
function r = invQumulFunc_r2(q, R)
% const radial pdf
q(q<0) = 0;
q(q>1) = 1;
alpha = 2/R^2;
r = (2*q/alpha).^(1/2);
end