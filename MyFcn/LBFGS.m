function [wcur,error,Obj,spentTime,trainError,testError]=LBFGS(X,Y,lambda,Xtest,Ytest,wcur,budget)
% minimizes: 1/n*sum i L(y i,W*x i) + lambda sum i sum j w fijg^2
% where L is the softmax/cross-entropy loss
% input: X should have dimension dim * N, where dim is the number of
% features and n the number of data points
% Y contains the label vector of size N (K different classes)
% lambda is the regularization parameter
% Xtest,Ytest is the training data (same format
% wcur = initialization - if not available set zeros(K,dim)
% budget = the budget in the limited BFGS method (number of stored
% gradients
% version - controls the matrix G fk-budgetg has to be either 0 (rescaled identity matrix -
% recommended), 1 (identity matrix)
[dim,N]=size(X); % dimension and number of datapoints
K=length(unique(Y)); % number of classes
valsLabel=unique(Y);
% label encoding
outY=zeros(K,N);
for i=1:K
ixx= find(Y==valsLabel(i));
outY(i,ixx)=1;
end
maxstepsize=1;
eps=5E-12*K*dim;
prodXw=wcur*X;
[CurObj,error,errorTest] = EvalObj(prodXw,wcur,X,Y,valsLabel,K,lambda,Xtest,Ytest);
tstart=tic;
spentTime=0;
Obj=CurObj;
gradObj = zeros(K,dim);
gradObjOld = gradObj;
gradObj = EvalGradObj(prodXw,wcur,X,Y,valsLabel,K,outY,lambda);
r=reshape(gradObj',K*dim,1);
counter=1; FLAG=1; sigma=0.1; betaFactor=0.5; clear beta;
while(FLAG & counter<2000)
% the descent direction r is computed below using LBFGS
descent = - r;
% reshape descent direction to dim*K matrix
descent=reshape(descent,dim,K);
descent=descent';
wold=reshape(wcur',K*dim,1);
normGrad=norm(gradObj);
if(normGrad>eps)
% do descent step
[prodXw,wcur,newObj,stepsize,error,errorTest]=getStepSize(wcur,X,Y,valsLabel,K,outY,sigma,betaFactor,CurOdisp(['Iteration: ',num2str(counter),' Stepsize: ',num2str(stepsize),' - NewObj: ',num2str(newObj,'%1.15f'),'- NormGrad: ',num2str(normGrad,'%1.15f')]);
disp(['Current Training error: ',num2str(error),' - Test error: ',num2str(errorTest)]);
%maxstepsize=min(stepsize*(1/betaFactor),1);
CurObj=newObj;
trainError(counter)=error; testError(counter)=errorTest;
spentTime=[spentTime, toc(tstart)];
Obj=[Obj, CurObj];
else
FLAG=0;
end
gradObjOld = gradObj;
gradObj = EvalGradObj(prodXw,wcur,X,Y,valsLabel,K,outY,lambda);
wnew=reshape(wcur',K*dim,1); wold=reshape(wold',K*dim,1);
sk = wnew-wold; yk=reshape((gradObj-gradObjOld)',K*dim,1);
rhok = 1/sum(sk.*yk);
%bk = G*yk;
descent=reshape(gradObj',K*dim,1);
if(counter<=budget)
s(:,counter)=sk; z(:,counter)=yk; rho(counter)=rhok;
for i=counter:-1:1
alpha(i)=rho(i)*sum(s(:,i).*descent);
descent = descent-alpha(i)*z(:,i);
end
if(version==1)
r=descent;%*(s(:,counter)'*z(:,counter))/(sum(z(:,counter).^2)); % this is the step where one could use a different H 0
else
r=descent*(s(:,counter)'*z(:,counter))/(sum(z(:,counter).^2));
end
for i=1:counter
beta = rho(i)*z(:,i)'*r;
r=r+s(:,i)*(alpha(i)-beta);
end
last=counter;first=1;
order=1:counter;
else
order=order-1; current=find(order==0); order(current)=budget;
s(:,current)=sk; z(:,current)=yk; rho(current)=rhok;
for i=budget:-1:1
index=find(order==i);
alpha(index)=rho(index)*sum(s(:,index).*descent);
descent = descent-alpha(index)*z(:,index);
end
if(version==1)
r=descent;%*(s(:,current)'*z(:,current))/(sum(z(:,current).^2)); % this is the step where one could use a different H 0
else
r=descent*(s(:,current)'*z(:,current))/(sum(z(:,current).^2)); % this is the step where one could use a different H 0
end
for i=1:budget
    index= find(order==i);
    beta = rho(index)*sum(z(:,index).*r);
    r=r+s(:,index)*(alpha(index)-beta);
end
end
%T = (eye(dim*K)-rho*sk*yk')*G*(eye(dim*K)-rho*yk*sk') + rho*sk*sk';
%G = G - rhok*(sk*bk'+bk*sk')+rhok*( sk*(1+rhok*yk'*bk)*sk');
%disp(['Check: ',num2str(sum(sum(abs(T-G))),'%1.15f')]);
counter=counter+1;
end
function [Objval,error,errorTest] = EvalObj(f,w,X,Y,valsLabel,K,lambda,Xtest,Ytest)
%f=w*X;
g=f-ones(K,1)*max(f);
loss=0;
for r=1:K
loss=loss-sum(g(r,Y==valsLabel(r)));
end
loss=loss+sum(log(sum(exp(g),1)));
loss=loss/length(Y);
Objval=loss+lambda/2*sum(w(:).^2);
% compute error
[ ,ix]=max(f,[],1);
error=sum(valsLabel(ix)==Y);
ftest=w*Xtest;
[ ,ix]=max(ftest,[],1);
errorTest=sum(valsLabel(ix)==Ytest);
function gradObj = EvalGradObj(f,w,X,Y,valsLabel,K,outY,lambda)
%f=w*X;
fnorm = f - ones(K,1)*max(f); % subtract for each column the maximum
expfnorm = exp(fnorm);
sumexpfnorm = sum(expfnorm);
softmax = expfnorm./(ones(K,1)*sumexpfnorm);
ObjGrad = softmax-outY; % softmax +sum of all function values
gradObj = (ObjGrad*X')/length(Y)+lambda*w;
function [totProd,wnew,newObj,stepsize,error,errorTest]=getStepSize(w,X,Y,valsLabel,K,outY,sigma,beta,oldObj)
t=maxstepsize; FLAG=1;
Product=sum(sum(gradObj.*descent));
disp(['Product: ',num2str(Product,'%1.15f')]);
prodXw=w*X;
prodXd=descent*X;
while(FLAG)
    wnew = w + t*descent;
    totProd=prodXw+t*prodXd;
    [newObj,error,errorTest]= EvalObj(totProd,wnew,X,Y,valsLabel,K,lambda,Xtest,Ytest);
    if( newObj > oldObj + sigma*t*Product)
        t=beta*t;
    else
        FLAG=0;
    end
end
stepsize=t;