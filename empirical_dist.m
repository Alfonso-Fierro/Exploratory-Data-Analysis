function ratio = empirical_dist(data, value)
%EMPIRICAL_DIST Calcular proporción de datos <= value
%   Dado un array, devolver la propor
    Ind = sum(data <= value);
    ratio = Ind/(size(data,1));
end

