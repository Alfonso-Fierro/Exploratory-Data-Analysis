P1= randn(1000,3);
%plot3(P(:,1), P(:,2), P(:,3), 'o')
P2 = [3 -3 3]+randn(100,3);
P3= [5 -5 5]+randn(100,3);
S =[P1
       P2 
          P3];
plot3(S(:,1), S(:,2), S(:,3), 'o')
vnorm        = normalize(S,2);
normalizados = transpose(vnorm);

matproyec = S*normalizados;
B = abs(kurtosis(matproyec));
MAX = max(B);
II= find(B==MAX); %La mayor proyeccion
XPP = matproyec(:,II);
histogram(XPP);
boxplot(XPP)
ri2 = iqr(XPP); %rango intercuantil donde termina el bigote
QQ3= prctile(XPP,75);
LL= QQ3+1.5*ri2;
II2= find(XPP>LL);
plot3(S(:,1), S(:,2), S(:,3), 'o')
hold on
plot3(S(II2,1),S(II2,2),S(II2,3),'or')