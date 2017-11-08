% script faster projected gradient
rng(2);
addpath /home/yu/bcd/BCD/
% input
l=0;u=1;
lambda=0.5;
d=100;
e1 = ones(d,1);
A = spdiags([-e1,-e1],[-1,1],d,d);
diagonal = -sum(A);
A = spdiags(diagonal'+lambda,0,A);

% load /home/yu/datasets/SNAP/Social_networks/mat/facebook_combined.mat
% d=size(A,1);
% diag=sum(A);
% A=A+speye(d)*0.5;


%xmin = rand(d,1)*2-0.5;
xmin = rand(d,1)*0.5+0.25;
b=A*xmin;
iter=200;
init=0.5;

[x,fx]=FPG(A,b,init,iter);


% CBCD
[cx1, cy1] = CBCD_size1_fx(A, b, d, iter,1E-10,0,1,init);
[cx2, cy2] = CBCD_size2_fx(A, b, d, iter,1E-10,0,1,init);
[cx3, cy3] = CBCD_size3_fx(A, b, d, iter,1E-10,0,1,init);

% plot
figure(1),clf,
fmin=min([fx;cy1;cy2;cy3]);
semilogy(1:length(fx),fx-fmin,'m','LineWidth',2);hold on;
semilogy(1:length(cy1),cy1-fmin,'r','LineWidth',2);hold on;
semilogy(1:length(cy2),cy2-fmin,'g','LineWidth',2);hold on;
semilogy(1:length(cy3),cy3-fmin,'b','LineWidth',2);hold on;

