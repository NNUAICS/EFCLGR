clc;
clear;%,
warning('off')
datasets_name={'isolet_uni',};
tools_name={'EFCLGR'};
for ds=1:length(datasets_name)
%     dataset=strrep(datasets_name(ds).name,'.mat',''); 
      dataset=datasets_name{ds};  
      file=strcat('D:\Matlab\DM\data\data\',dataset,'.mat'); 
    load(file);
    for drt=1:length(tools_name)
        tool=tools_name{drt};
        filename=strcat('D:\Matlab\DM\LPPFCM\EFCLGR\result\',dataset,"_",tool,".xlsx");
        if exist(filename,'file')
           continue;
        end


        class=length(unique(Y));

        result=[];
        measure=[];

        measureCount=30;
        if size(X,2)>100
            X=double(X);
            data=pca(X,100);
            [n,d]=size(data);
            strat_dim=10;
            step=10;
            end_dim=100;
        end
            X = data;
        for dim=strat_dim:step:end_dim
            dim%.......................
            if strcmp(tool,"EFCLGR")
                    for sigma=[0.1,1,10]
                            [L,C] = compute_L(X,sigma);
                                for lamda=[0.1,1,10]
                                    for gamma =[0.1,1,10]
                                        for theta=[0.1,1,10]
                                            for mu=[100]
                                                for rho=1.01
                                                    ACCs=[];
                                                   NMIs=[];
                                                    PURITYs=[];
                                                    for a=1:1:30
                                                    [U] = EFCLGR(X,L,C,class,dim,lamda,gamma,theta,mu,rho);

                                                    [max_a,index]=max(U,[],2);
                                                    
                                                    [ACC,MIhat,Purity]=ClusteringMeasure(Y',index');
                                                  % ACC

                                                     ACCs=[ACCs;ACC];
                                                     NMIs=[NMIs;MIhat];
                                                     PURITYs=[PURITYs;Purity];
%                                                     result=[result;ACC,MIhat,Purity,sigma,lamda,gamma,theta,mu,rho,dim];
                                                     end
                                                      
                                                       [mean_measure,std_measure]=ud_measure(ACCs,NMIs,PURITYs,measureCount);
                                                 mean_measure
                                                      result=[result;mean_measure,std_measure,sigma,lamda,gamma,theta,mu,rho,dim];
                                                     
                                               
                                                end
                                            end
                                        end   
                                    end
                                end
                            end
                        end
                                  
            end
%    end
%         
   if exist(filename,'file')%存在则删除
            delete(filename)
   end
    end
  xlswrite(filename,result)
            
%          end
end


%%
% clustering
function [mean_measure,std_measure]=ud_measure(ACC,MIhat,Purity,iters)
measure=[];
for count=1:iters
    measure=[measure;ACC,MIhat,Purity]; %#ok<AGROW>
end
mean_measure=mean(measure,1);
std_measure=std(measure,1);
end
%%
function [mean_measure,std_measure]=udran_measure(data,target,num_samp,iters)
measure=[];
for count=1:iters
    [~,~,idx_train]=Random_sampling(target,num_samp,'class');
    idx_test=setdiff(1:length(target),idx_train);
    mdl=fitcknn(data(idx_train,:),target(idx_train),'NumNeighbors',1,'Distance','euclidean');
    pred_label=predict(mdl,data(idx_test,:));
    [out] = classification_evaluation(target(idx_test)',pred_label');
    measure=[measure;out.avgAccuracy,out.fscoreMacro,out.fscoreMicro];
end
mean_measure=mean(measure,1);
std_measure=std(measure,1);
end
%%
function [mean_measure,std_measure]=new_measure(data,class,target,num_count)
measure=[];
for count=1:num_count
    [index,~,~] = kmeans(data,class);
    [ACC,MIhat,Purity]=ClusteringMeasure(target',index');
    measure=[measure;ACC,MIhat,Purity];
end
mean_measure=mean(measure,1);
std_measure=std(measure,1);
end
