function [x,fx]=FPG(A,b,init,acc,iter)
% nesterov's faster projected gradient descent
% output
lower = 0;
upper = 1;
x = ones(size(b))*init;
fx= zeros(iter,1)*inf;
y = x;
fx(1)=f(A,b,x);

beta=0.8;
sigma=0.8;
tmax=1;
KKT=1;
i=1;
t=1/max(svds(A));
while KKT>acc && i<iter
    x0= x;
    d = A*y-b;
    %t=getStepSize(A,b,x,-d,tmax,beta,sigma);
    x = max(min(y-t*(d),1),0);
    y = x + i/(i+3)*(x-x0);
    fx(i+1)=f(A,b,x);
    i=i+1;
    % show the KKT condition
    if mod(i,100)==0
        % compute the real gradient after each epoch
        grad = A*x;
        % opt condition, 0 in sub gradient
        index_l = find(x<=lower+2*eps);
        index_u = find(x>=upper-2*eps);
        index = find(x>lower+2*eps & x<upper-2*eps);
        KKT = norm([grad(index)-b(index);min(0,grad(index_l)-b(index_l));...
            max(0,grad(index_u)-b(index_u))],2);
        fprintf('i=%d; KKT=%f\n',i,KKT);
    end
end
fx(isnan(fx))=[];
end

function y=f(A,b,x)
y=0.5*x'*A*x-x'*b;
end

function t=getStepSize(A,b,x,d,t,beta,sigma)
% backtracking line search
FLAG=1;
while(FLAG)
    f=0.5*x'*A*x-b'*x;
    xnew = x+t*d;
    fnew=0.5*xnew'*A*xnew-b'*xnew;
    if( fnew-f >sigma*t*sum((A*x-b).*d))
        t=beta*t;
    else
        FLAG=0;
    end
end

%t=1;
end
