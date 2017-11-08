function [xcur,f,q]=BFGS(A,b,xcur,budget)
% min 1/2<x,Ax>-<b,x>
% unconstrained problem


% parameters of getStepSize
beta_=0.95;
sigma_=0.45;
tmax=10;

d = size(A,1);
% allocate parameters
max_iter=100;
f=zeros(max_iter,1)+inf;
s = zeros(d,budget);
y = s;
alpha = zeros(budget,1);
beta = alpha;
rho  = alpha;

k=0;
gradOld = A*xcur-b;
% D_0=I; so the first descent is:-I*gradOld
dk = -gradOld;
% get step size
lambda = getStepSize(A,b,xcur,dk,tmax,beta_,sigma_);%%%%
% update rho, s, y
sk = lambda*dk;
s(:,1)=sk;
xnew = xcur+sk;
q = A*xnew-b;
yk = q-gradOld;
y(:,1)=yk;
xcur=xnew;gradOld=q;
rho(1)=1/sum(yk.*sk);
f(1)=0.5*xcur'*A*xcur-b'*xcur;
disp(['k:' num2str(k+1), ' -Stepsize:',num2str(lambda,'%1.5f'), ' -F:',...
    num2str(f(1),'%1.10f'), ' -NormGrad:' num2str(norm(q),'%1.15f')]);
k=1;

while norm(gradOld)>1E-10 && k<2000 && lambda>1E-100
    % step 1
    if k<=budget
        L=k;
    else
        L=budget;
    end
    % step 2
    for j=L:1
        alpha(j)= rho(j)*sum(q.*s(:,j));
        q=q-alpha(j)*y(:,j);
    end
    % step 3
    r = sum(sk.*yk)/sum(yk.*yk)*q;
    for j=1:L
        beta(j)=rho(j)*sum(r.*y(:,j));
        r=r+(alpha(j)-beta(j))*s(:,j);
    end
    % now we have r=D*g
    dk=-r/norm(r);
    % get step size
    lambda=getStepSize(A,b,xcur,dk,tmax,beta_,sigma_);%%%%
    % update rho, s, y
    k=k+1;
    sk = lambda*dk;
    xnew = xcur+sk;
    q = A*xnew-b;
    yk = q-gradOld;
    if k<=budget
        s(:,k)=sk;
        y(:,k)=yk;
        rho(k)=1/sum(yk.*sk);
    else
        % remove the first colomn
        s(:,budget+1)=sk;
        s(:,1)=[];
        y(:,budget+1)=yk;
        y(:,1)=[];
        rho(budget+1)=1/sum(yk.*sk);
        rho(1)=[];
    end
    xcur=xnew;gradOld=q;
    fval=0.5*xcur'*A*xcur-b'*xcur;
    f(k)=fval;
    if mod(k,10)==0
        disp(['k:' num2str(k), ' -Stepsize:',num2str(lambda,'%1.5f'), ' -F:',...
            num2str(fval,'%1.10f'), ' -NormGrad:' num2str(norm(q),'%1.15f')]);
    end
end
disp(num2str(lambda));

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



