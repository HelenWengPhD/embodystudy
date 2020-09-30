function [acts scratchpad] = test_L2_RLR( testpats, testtargs, scratchpad )

% Generates predictions (acts) using a trained L2 regularized logistic regression model
%
% [ACTS SCRATCHPAD] = test_L2_RLR(TESTPATS,TESTTARGS,SCRATCHPAD)


examples = testpats'; nExamples = size(examples,1);
acts = exp( examples * scratchpad.weights + repmat(scratchpad.biases,nExamples,1) )' ./ (1+exp( examples * scratchpad.weights + repmat(scratchpad.biases,nExamples,1) )');