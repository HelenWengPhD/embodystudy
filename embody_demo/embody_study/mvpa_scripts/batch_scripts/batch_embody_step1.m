function P = batch_embody_step1()
% function to run step1_machinelearning.m for all subjects

% Edit path to local EMBODY study root directory
root_dir = 'Path/to/EMBODY/root/directory';

% Add all subject IDs (str)
subjID = {'124'};

% All subject orders (str)
% Subj order must be matched to subject number order
orderSet = {'order1', 'order3', 'order2'};

% label is name of folder within EMBODY study root directory 
label = 'phase1_demo';
mapObj = containers.Map(subjID, orderSet);
brainproc = 'tamd';
maskName = 'INTRACRANIAL_tam';
shiftTRs = 6;
PENALTY = {.01};
featSel = 0;

for j = 1:length(PENALTY)
  for i = 1:length(subjID) 
  
  order = mapObj(subjID{i});
  step1_machinelearning(root_dir, subjID{i}, order, label, brainproc, maskName, shiftTRs, PENALTY{j}, featSel)

  end
end
