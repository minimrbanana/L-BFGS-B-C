%script lbfgs-b

disp('=== My test function, 2D === ');
load /home/yu/datasets/SNAP/Social_networks/mat/facebook_combined.mat
d=size(A,1);
diag=sum(A);
A=A+speye(d)*1E-6;
rng(1);
x_true=rand(d,1)*0.8+0.1;
b=A*x_true;
fmin=0.5*x_true'*A*x_true-b'*x_true;
n = d;

f = @(x) 0.5*x'*A*x-b'*x;
g = @(x) A*x-b;

% There are no constraints
l   = -inf(n,1);
u   = inf(n,1);

opts    = struct( 'x0', ones(d,1)*0.5 );
opts.printEvery     = 1;
opts.m  = 5;

% Here's an example of using an error function. For Rosenbrock,
%   we know the true solution, so we can measure the error at every
%   iteration:
trueSoln = x_true;
% "errFcn" will be printed to the screen
opts.errFcn     = @(x) norm(x-trueSoln)/max(norm(trueSoln),1);
% "outputFcn" will save values in the "info" output
opts.outputFcn  = opts.errFcn;

% Ask for very high accuracy
opts.pgtol      = 1e-10;
opts.factr      = 1e3;

% The {f,g} is another way to call it
[x,f,info] = lbfgsb( {f,g} , l, u, opts );

if abs(f) < 1e-8
    disp('Success!');
% since we included opts.outputFcn, the info.err now has 3 columns.
%   The first 2 columns are the same as before; the 3rd column
%   is the output of our outputFcn
semilogy( info.err(:,3)-f,'o-' ); xlabel('iteration'); ylabel('relative error in iterate function');
else
    disp('Something didn''t work right :-(  ');
end

figure(1),
fval=info.err(:,1);
semilogy(1:length(fval),fval-fmin,'r','LineWidth',2.5);hold on;