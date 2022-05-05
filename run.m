clear;
clc;

addpath(genpath('./'));

resultdir1 = 'Results/';
if (~exist('Results', 'file'))
    mkdir('Results');
    addpath(genpath('Results/'));
end

resultdir2 = 'aResults/';
if (~exist('aResults', 'file'))
    mkdir('aResults');
    addpath(genpath('aResults/'));
end

datadir='./datasets/';

dataname={'ORL_mtv'};

numdata = length(dataname); % number of the test datasets
numname = {'_Per0.1', '_Per0.2', '_Per0.3', '_Per0.4','_Per0.5', '_Per0.6', '_Per0.7', '_Per0.8', '_Per0.9'};

for idata = 1:1 :length(dataname)
    ResBest = zeros(9, 8);
    ResStd = zeros(9, 8);
    for dataIndex = 1:1:9

        datafile = [datadir, cell2mat(dataname(idata)), cell2mat(numname(dataIndex)), '.mat'];
        load(datafile);
        %data preparation...
        gt = truelabel{1};
        cls_num = length(unique(gt));
        k= cls_num;
        tic;
        [X1, ind] = findindex(data, index);

        
        time1 = toc;
        maxAcc = 0;
        TempLambda1 = [0.001 0.1 1 10 100 1000];
        TempLambda2 = [k,2*k, 3*k,5*k,7*k];
        
        ACC = zeros(length(TempLambda1),length(TempLambda2));
        NMI = zeros(length(TempLambda1), length(TempLambda2));
        Purity = zeros(length(TempLambda1), length(TempLambda2));
        idx = 1;
        for LambdaIndex1 = 1 : length(TempLambda1)
            lambda1 = TempLambda1(LambdaIndex1);
            for LambdaIndex2 = 1 : length(TempLambda2)
                lambda2 = TempLambda2(LambdaIndex2);
                disp([char(dataname(idata)), char(numname(dataIndex)), '-l1=', num2str(lambda1), '-l2=', num2str(lambda2)]);
                tic;
                para.c = cls_num; % K: number of clusters
                para.k = lambda1; % m: number of nearest anchors
                [F,V,A,Z,iter,obj] = missingalgo_qp(X1,gt,lambda1,lambda2,ind); % X,Y,lambda,d,numanchor
                
                F = F ./ repmat(sqrt(sum(F .^ 2, 2)), 1, k);
                
                time2 = toc;
                stream = RandStream.getGlobalStream;
                reset(stream);
                MAXiter = 1000; % Maximum number of iterations for KMeans
                REPlic = 20; % Number of replications for KMeans
                tic;
                for rep = 1 : 20
                    pY = kmeans(F, cls_num, 'maxiter', MAXiter, 'replicates', REPlic, 'emptyaction', 'singleton');
                    res(rep, : ) = Clustering8Measure(gt, pY);
                end
                time3 = toc;
                runtime(idx) = time1 + time2 + time3/20;
                disp(['runtime:', num2str(runtime(idx))])
                idx = idx + 1;
                tempResBest(dataIndex, : ) = mean(res);
                tempResStd(dataIndex, : ) = std(res);
                ACC(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 1);
                NMI(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 2);
                Purity(LambdaIndex1, LambdaIndex2) = tempResBest(dataIndex, 3);
                save([resultdir1, char(dataname(idata)), char(numname(dataIndex)), '-l1=', num2str(lambda1), '-l2=', num2str(lambda2), ...
                    '-acc=', num2str(tempResBest(dataIndex,1)), '_result.mat'], 'tempResBest', 'tempResStd');
                for tempIndex = 1 : 8
                    if tempResBest(dataIndex, tempIndex) > ResBest(dataIndex, tempIndex)
                        if tempIndex == 1
                            newZ = Z;
                            newF = F;
                        end
                        ResBest(dataIndex, tempIndex) = tempResBest(dataIndex, tempIndex);
                        ResStd(dataIndex, tempIndex) = tempResStd(dataIndex, tempIndex);
                    end
                end
            end
        end
        aRuntime = mean(runtime);
        PResBest = ResBest(dataIndex, :);
        PResStd = ResStd(dataIndex, :);
        save([resultdir2, char(dataname(idata)), char(numname(dataIndex)), 'ACC_', num2str(max(ACC(:))), '_result.mat'], 'ACC', 'NMI', 'Purity', 'aRuntime', ...
            'newZ', 'newF', 'PResBest', 'PResStd');
    end
    save([resultdir2, char(dataname(idata)), '_result.mat'], 'ResBest', 'ResStd');
end
