function [UU,V,A,Z,iter,obj] = missingalgo_qp(X,Y,lambda,numanchor,ind)
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.

% X      : n*di

%% initialize
maxIter = 50 ; % the number of iterations

m = numanchor;
numclass = length(unique(Y));
numview = length(X);
numsample = size(Y,1);
Z = zeros(m,numsample); 
Z(:,1:m) = eye(m);

missingindex = constructA(ind);
for i = 1:numview
    di = size(X{i},1); 
    A{i} = zeros(di,m); 
end

alpha = ones(1,numview)/numview;

flag = 1;
iter = 0;
%%
while flag
    iter = iter + 1;

    %% optimize Ai
    for ia = 1:numview
        part1 = X{ia} * Z';
        [Unew,~,Vnew] = svd(part1,'econ');
        A{ia} = Unew*Vnew';
    end
    
    %% optimize Z
    C1 = 0;
    C2 = 0;
    for a=1:numview
        C1 = C1 + alpha(a)^2*ind(:,a)'; 
        C2 = C2 + alpha(a)^2 * A{a}'*X{a};
    end
    C1 = C1 + lambda * ones(1,numsample);
    for ii=1:numsample
        idx = 1:numanchor;
        ut = C2(idx,ii)./C1(ii);
        Z(idx,ii) = EProjSimplex_new(ut');
    end

    %% optimize alpha
    M = zeros(numview,1);
    for iv = 1:numview
        M(iv) = norm( X{iv} - A{iv} * (Z.*repmat(missingindex{iv},m,1)),'fro')^2;
    end
    Mfra = M.^-1;
    Q = 1/sum(Mfra);
    alpha = Q*Mfra;

    %%
    term1 = 0;
    term2 = 0;
    for iv = 1:numview
        term1 = term1 + alpha(iv)^2 * norm(X{iv} - A{iv} * (Z.*repmat(missingindex{iv},m,1)),'fro')^2;
    end
    term2 = lambda * norm(Z,'fro')^2;
    obj(iter) = term1+ term2;
    
    
    if (iter>1) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-3 || iter>maxIter || obj(iter) < 1e-10)
        [UU,~,V]=svd(Z','econ');
        UU = UU(:,1:numclass);
        flag = 0;
    end
end
         
         
    
