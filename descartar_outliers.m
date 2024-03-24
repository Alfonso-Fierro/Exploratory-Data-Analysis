t1= randn(100,4)
%plot3(P(:,1), P(:,2), P(:,3), 'o')
t2 = [3 -3 3 -3]+exprnd(18,100,4)
t3= [5 -5 5 -5]+randn(100,4)
t4=[7 -7 7 -7]+exprnd(7,100,4)
T =[t1
       t2 
          t3
             t4]
vnorm2=normalize(T);
normalizados2=transpose(vnorm2);

matproyec2=T*normalizados2;
B= abs(skewness(matproyec2));
MAX = max(B);
II= find(B==MAX); %La mayor proyeccion
XPP = matproyec2(:,II);
histogram(XPP);
boxplot(XPP);
length(XPP);
treal=length(XPP);
XPP = sort(XPP);

while length(XPP)>treal*0.9
    ri2 = iqr(XPP); %rango intercuantil donde termina el bigote
    QQ3= prctile(XPP,75); %tener percentiles
    QQ1= prctile(XPP,25);
    LU= QQ3+1.5*ri2;    %Bigote arriba
    LL= QQ1-1.5*ri2;    %Bigote abajo
    
    upper = max(XPP);
    lower = max(XPP);
    %si no hay nada por fuera de los bigotes, sale del while
    if (upper < LU) && (lower > LL)
        break;
    end
    
    if LU < upper
        XPP(end) = [];
    end

    if LL > lower
        XPP(1) = [];
    end
end

length(XPP)
histogram(XPP)
boxplot(XPP)
ri2 = iqr(XPP); %rango intercuantil donde termina el bigote
QQ3= prctile(XPP,75) ;
LL= QQ3+1.5*ri2;
II2= find(XPP>LL);
histogram(XPP)
boxplot(II2)