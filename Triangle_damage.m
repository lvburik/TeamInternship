function [damage, inner_edges] = Triangle_damage
    %damage dimension
    length = 0.03;
    offset_x = -0.0;
    offset_y = -0.0;
    inner_edges = [3,4,7]; % inner edges to add face
    
    %define corners of polyshape
    damage_x = [-1 1 0]*length/2 + offset_x;
    damage_y = [0 0 2]*length/2 + offset_y;
    
    %load damage polyshape
    damage = polyshape(damage_x, damage_y);
end