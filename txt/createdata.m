

Nx=20;
Ny=20;
nb=2*Nx*Ny;

X=load('X.dat');
Ndata=size(X,1);
nh=Nx*Ny;


sets=size(X,1);

y=ones(Ndata,1);
y(1:Ndata/2)=0;


%X=[y,X];

X=uint8(X);
y=uint8(y);


%save('extended.mat','X','y');

%fwrite('ising.bin',X)

dlmwrite('Xtrain.txt', X,'delimiter', ' ')
dlmwrite('ytrain.txt', y,'delimiter', ' ')


X=load('Xtest.dat');
Ndata=size(X,1);
nh=Nx*Ny;


sets=size(X,1);

y=ones(Ndata,1);
y(1:Ndata/2)=0;


%X=[y,X];

X=uint8(X);
y=uint8(y);


%save('extended.mat','X','y');

%fwrite('ising.bin',X)

dlmwrite('Xtest.txt', X,'delimiter', ' ')
dlmwrite('ytest.txt', y,'delimiter', ' ')

