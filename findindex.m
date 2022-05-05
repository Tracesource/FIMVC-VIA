function [X1, ind] = findindex(data, index)
%FINDINDEX Summary of this function goes here
%   Detailed explanation goes here
[numofview,~] = size(data);
[~,numofsample] = size(data{1});

X1 = cell(numofview,1);

ind = zeros(numofsample,numofview);
for i=1:numofview
    [d,~]=size(data{i});
    ind(index{i}, i) = 1;
    origin = data{i};
    origin(isnan(origin)) = 0;
    X1{i} = NormalizeData(origin);
end




end

