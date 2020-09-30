function [scratchpad] = train_L2_RLR(trainpats,traintargs,class_args,cv_args)

%% Train a K class logistic regression classifier, with optional regularization via L2 norm of weight vector(s)

% To implement L2 regularization, user must specify class_args.lambda parameter



%% process arguments
                         
% default parameters
defaults.regularization = 'L2';
defaults.lambda         = 1;
defaults.optimization   = 'minimize';
defaults.labelsGroup    = []; 


 class_args = mergestructs(class_args, defaults);

  regularization = class_args.regularization;
  lambda         = class_args.lambda;
  optimization   = class_args.optimization;
  lambda         = class_args.penalty;
  
%% call the classifier function

model = classifierLogisticRegression( trainpats', traintargs', 'lambda', lambda);

%% pack the results

scratchpad.W = model.W;
scratchpad.weights = model.weights;
scratchpad.biases  = model.biases;

scratchpad.lambda         = lambda;
scratchpad.regularization = regularization;
scratchpad.optimization   = optimization;
