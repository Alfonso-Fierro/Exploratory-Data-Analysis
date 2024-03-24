%Test de Independencia para copulas
%Iterar en la matríz de ECDF
%Check sobre la matríz  x menor que x || y menor que y
%guardar (CHECK , X*Y)
%Mostrar regresión lineal

function indep = test_copula(datx,daty)
    ecdv1 = zeros(size(datx));
    ecdv2 = zeros(size(datx));
    
    for i = 1:size(datx,1)
        ecdv1(i) = empirical_dist(datx, datx(i));
    end
    
    for i = 1:size(var2,1)
        ecdv2(i) = empirical_dist(daty, daty(i));
    end
end
