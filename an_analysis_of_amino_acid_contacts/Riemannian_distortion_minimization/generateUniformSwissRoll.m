function [data, color, s] = generateUniformSwissRoll(th0, thf, z0, zf, N, holeRange)
% holeRange: [xleft, xright; ybottom, ytop]
if nargin < 6
    holeRange = [];
end
s0 = swissRoll_th2s(th0);
sf = swissRoll_th2s(thf);
if isempty(holeRange)
    s = s0 + (sf - s0)*rand(1,N);
    z = z0 + (zf - z0)*rand(1,N);
else
    Ntemp = fix(2*N/(1-(holeRange(1,2) - holeRange(1,1))*(holeRange(2,2) - holeRange(2,1))));
    s = s0 + (sf - s0)*rand(1,Ntemp);
    z = z0 + (zf - z0)*rand(1,Ntemp);
    temp = and(and(s < s0 + (sf - s0)*holeRange(1,2), s > s0 + (sf - s0)*holeRange(1,1)), ...
        and(z < z0 + (zf - z0)*holeRange(2,2), z > z0 + (zf - z0)*holeRange(2,1)));
    s = s(~temp);
    z = z(~temp);
    if length(s) < N
        N = length(s);
    end
    s = s(1:N);
    z = z(1:N);
end
color = (s-s0)/(sf-s0) - 0.5;
th = zeros(1,N);
for i=1:N
    th(i) = swissRoll_s2th(s(i));
end
data = [th.*cos(th); th.*sin(th); z];
data = data';
color = color';