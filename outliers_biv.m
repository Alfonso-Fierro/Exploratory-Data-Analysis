%Detección de Outlayers multivariados. Método de máxima varianza

%Forma Bi-Variada. %Generar vectores de proyección con coordenadas
%esféricas

angles = linspace(0, 2*pi, 1100); %recorrido del círculo unitario
y_values = sin(angles)';
x_values = cos(angles)';
% size(x_values) revisar dimension de valores de x o y
vectors = [y_values, x_values]; %formar vectores
% size(vectors) revisar dimension de vectores

%el producto debe reducir la dimension de n x 2  a n x 1

X1 = data5(:,1); % normal estandar con independencia. Es una esfera
X2 = data5(:,3);
X  = X1:X2;
plot(X(:,1),X(:,2),'o')
size(X)
size(vectors)
P = X*vectors'; % Esta ya es la matriz multivariante. Se pasó de datos multivariantes a univariantes
%plot(x,y,'o')
A = abs(var(P));
MA = max(A); %Indices con máxima asimetría
I= find(A==MA); %La mayor proyeccion es criterio para detectar outliers
XP = P(:,I); %La proyeccion mayor en la matriz de proyecciones
histogram(XP);
boxplot(XP);
ri = iqr(XP); %rango intercuantil donde termina el bigote
Q3= prctile(XP,75);
L= Q3+1.5*ri;
I2= find(XP>L);
plot(X(:,1), X(:,2),'o')
hold on
plot(X(I2,1), X(I2,2),'or')
hold off
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


