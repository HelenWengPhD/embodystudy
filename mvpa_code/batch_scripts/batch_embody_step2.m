function P = batch_embody_step2()
% function to run step2_machinelearning.m for all subjects

% Edit path to local EMBODY study root directory
root_dir = 'Path/to/EMBODY/root/directory';

% Add all subject IDs (str)
subjID = {'100', '101', '102'};

% All subject orders (str)
% Subj order must be matched to subject number order
orderSet = {'order1', 'order3', 'order2'};

% label is name of folder within EMBODY study root directory 
label = 'insert analysis name';
mapObj = containers.Map(subjID,orderSet);
brainproc = 'insert_fmri_preproc_filename';
maskName = 'insert_mask_name';
shiftTRs = 6;
PENALTY = {.01};
featSel = 0; % wholebrain analysis

for j = 1:length(PENALTY)
  for i = 1:length(subjID) 
  
  order = mapObj(subjID{i});
  step2_machinelearning(root_dir, subjID{i}, order, label, brainproc, maskName, shiftTRs, PENALTY{j}, featSel)

  end
end
