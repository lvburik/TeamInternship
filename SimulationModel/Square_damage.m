function [damage, inner_edges] = Square_damage
    %damage dimension
    length = 0.03;
    offset_x = -0.0;
    offset_y = -0.0;
    inner_edges = [2,3,7,8]; % inner edges to add face
    
    %define corners of polyshape
    damage_x = [-1 1 1 -1]*length/2 + offset_x;
    damage_y = [-1 -1 1 1]*length/2 + offset_y;
    
    %load damage polyshape
    damage = polyshape(damage_x, damage_y);
end
