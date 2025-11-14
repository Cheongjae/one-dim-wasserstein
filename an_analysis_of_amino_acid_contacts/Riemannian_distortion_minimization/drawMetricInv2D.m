function [points, V, D, points2] = drawMetricInv2D(Minv, pos, color, drawMinEigValDir)
if nargin < 4
    drawMinEigValDir = false;
    if nargin < 3
        color = 'r';
    end
end
points = [];
V = [];
D = [];
points2 = [];
if size(Minv,2) == 2
    [V, D] = eig(Minv);
    D = sqrt(D);
    nDraw = 100;
    theta = linspace(0,2*pi,nDraw);
    points = V*[(D(1,1))*cos(theta); (D(2,2))*sin(theta)] + pos*ones(1,nDraw);
    h = plot(points(1,:), points(2,:));
    set(h, 'color', color)
elseif size(Minv,2) == 3
    [V, D] = eig(Minv);
    d = sqrt(diag(D));
    nDraw1 = 20;
    nDraw2 = 10;
    theta = linspace(0,2*pi,nDraw1);
    phi = linspace(0,pi,nDraw2);
    [th, ph] = meshgrid(theta, phi);
    th = reshape(th, 1, size(th,1)*size(th,2));
    ph = reshape(ph, 1, size(ph,1)*size(ph,2));
    points = V*[(d(1))*cos(th).*sin(ph); (d(2))*sin(th).*sin(ph); (d(3))*cos(ph)] + pos*ones(1,nDraw1*nDraw2);
    h = plot3(points(1,:), points(2,:), points(3,:));
    set(h, 'color', color)
    
    % min eigen value direction
    if drawMinEigValDir
        [min_d, minIdx] = min(d);
        scalefactor = 10;
        points2 = scalefactor * min_d * V(:,minIdx)*[-1, 1] + pos;
        h2 = plot3(points2(1,:), points2(2,:), points2(3,:));
        set(h2, 'color', 'k')
    end
end