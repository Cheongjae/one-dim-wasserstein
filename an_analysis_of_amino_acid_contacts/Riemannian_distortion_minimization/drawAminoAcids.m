function drawAminoAcids(Y, markers, legends, titles, newfigure)
CELL = {'CYS', 'PHE', 'ILE', 'VAL', 'LEU', 'MET', 'TRP', 'HIS', 'TYR', 'ALA', 'GLY', 'THR'};
% CELL = {'CYS', 'PHE', 'ILE', 'VAL'};
if nargin < 5
    newfigure = true;
end
if newfigure
    figure
end
for i=1:20
    if markers(i) == '+' || markers(i) == '.' || markers(i) == 'x' || markers(i) == '*'
        p = scatter(Y(i,1),Y(i,2),45);
    else
        p = scatter(Y(i,1),Y(i,2),45,'filled');
    end
    p.Marker = markers(i);
    if any(strcmp(CELL,legends(i)))
        text(Y(i,1)+0.05, Y(i,2), legends(i))
    end
    hold on
end
title(titles)
legend(legends,'Position',[0.85 0.2 0.2 0.8])

% legend('Location','eastoutside')
max_norm = max(sqrt(sum(Y.*Y,2)));
stepsize = max_norm/5;
for i=1:5
    circle(0,0,stepsize*i);
end
axis equal
drawnow

end

function h = circle(x,y,r)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;
h = plot(xunit, yunit, 'k--', 'HandleVisibility','off');
hold off
end