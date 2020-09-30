% [] = EMBODY_MVPA(STAMP,SUBJNUM,NUMITERS,...)
%
% * STAMP      = date stamp for this analysis run
% * SUBJID     = subject ID string
% * ORDER      = order1-4 of IA task conditions and timings
% * FEATSEL    = (0/1) do voxelwise ANOVA feature selection on train data
% * VERBOSE    = (0/1) save all classification structures in addn to results file
% * SHIFTTRS   = # of TRs to shift data to adjust for hemodynamic lag
% * PENALTY    = (try 50 as a place to start)
% * NITERS     = # of random replications
% * BRAINPROC  = fmri preprocessed file name
% * MASKNAME   = name of mask to use to read in data
%
% Code developed by Jarrod Lewis-Peacock, Helen Weng, Mark Estefanos
% used in conjunction with MVPA Toolbox https://github.com/PrincetonUniversity/princeton-mvpa-toolbox

function [] = step2_machinelearning(subjID, order, label, brainproc, maskName, shiftTRs, PENALTY, featSel)
   
   nIters=1;
   verbose=1; 
   
%----------------------------------------------------------------------
% initialize subject structure with 'study' and 'subject' info

subj_stem = sprintf('embody_%s',subjID);
subj=init_subj('embody',subj_stem);
cd(sprintf('%s/%s/subjects/%s/', root_dir, label, subjID));
disp(sprintf('Subject directory: %s', pwd));

%----------------------------------------------------------------------
% create directory for output files
%

   output_dir = sprintf('%s/%s/results/step2/%s/',...
                     root_dir, label, subjID);
                 
   system(sprintf('mkdir -vp %s',output_dir));

%----------------------------------------------------------------------
                 
   diary([output_dir, 'log.txt']);

   disp(sprintf('Output directory: %s', output_dir));
   disp(sprintf('Subject number: %s', subjID));
   disp(sprintf('Order number: %s', order));
   disp(sprintf('Label: %s', label));
   disp(sprintf('Brain processed: %s', brainproc));
   disp(sprintf('Mask: %s', maskName));
   disp(sprintf('shift TR: %d', shiftTRs));
   disp(sprintf('Penalty: %f', PENALTY));
   disp(sprintf('Feature selection: %d', featSel));
                       
% [] = EMBODY_MVPA(STAMP,SUBJNUM,NUMITERS,...)
%
% * STAMP      = date stamp for this analysis run
% * SUBJID     = subject ID string
% * FEATSEL    = (0/1) do voxelwise ANOVA feature selection on train data
% * VERBOSE    = (0/1) save all classification structures in addn to results file
% * SHIFTTRS   = # of TRs to shift data to adjust for hemodynamic lag
% * PENALTY    = (try 50 as a place to start)
% * NITERS     = # of random replications
% * MASKNAME   = name of mask to use to read in data
%
% 1. read 8 blocks of data from embody study
% 2. do cross-validation of the data using 1 of 2 sets of regressors
%   (1) choose step1 model to train classifier
%   (2) decode meditation period
  
% read in voxels from the pre-processed files.
%

maskLoc = sprintf('%s+orig',maskName);
subj = load_afni_mask(subj,'read_mask',maskLoc);
  
disp('++ read in embody data');
raw_filenames={};
for i=1:8 % enter number of all runs (IA task + meditation)
  raw_filenames{i} = sprintf(['run%d_%s+orig'],i,brainproc);
  disp(sprintf('-+ %s', raw_filenames{i})); 
end
readmask = get_mat(subj,'mask','read_mask');
readsize = length(find(readmask));
disp(sprintf('++ loading data using "%s" with %d voxels',maskName,readsize));
subj = load_afni_pattern(subj,'epis','read_mask',raw_filenames);

%----------------------------------------------------------------------
% regressors & selectors
% 
% read in subject-specific stuff

regs_file = sprintf('%s/%s/scripts/regressors/%s/regressors.mat',...
                    root_dir, label, order);

load(regs_file)

subj = init_object(subj,'regressors','regs');
subj = set_mat(subj,'regressors','regs',regressors);

runs_file = sprintf('%s/%s/scripts/regressors/selector.mat', root_dir, label);
load(runs_file)
subj = init_object(subj,'selector','runs');
subj = set_mat(subj,'selector','runs',selector);

clear *selec*

%----------------------------------------------------------------------
% shift regressors to account for hemodynamic lag
%
subj = shift_regressors(subj,'regs','runs',shiftTRs);

%----------------------------------------------------------------------
% exclude rest timepoints
%
% need to exlude rest timepoints from the analysis
% want to set all 'baseline' timepoints to '0' so they are excluded
% create a new selector called 'no_rest' with rests points set to 0 
% 
regs = get_mat(subj,'regressors',sprintf('regs_sh%d',shiftTRs)); 
len = size(regs,2);
temp_sel = ones(1,len);
temp_sel(find(sum(regs)==0)) = 0;
subj = init_object(subj,'selector','runs_norest');
subj = set_mat(subj,'selector','runs_norest',temp_sel);
clear regs temp_sel len

%----------------------------------------------------------------------
% z-score
%
% It helps to zscore the data by subtracting out the mean of each
% voxel's timecourse and scaling it so that the standard deviation of
% the timecourse is one.
%
% Be sure to use the full 'runs' selector here so we include the 'rest' TRs
% in the zscoring
%
disp('++ z-scoring data');
subj = zscore_runs(subj,'epis','runs');

%----------------------------------------------------------------------
% create cross-validation indices
%
% We are going to create a group of selectors in anticipation of the
% n-minus-one cross-validation scheme that we will use to train and test our
% classifier.
%
subj = create_xvalid_indices(subj,'runs',...
                             'actives_selname','runs_norest');

if (nIters > 0)
  %----------------------------------------------------------------------
  % feature select anova
  %
  % Before actually doing the classification, it usually helps to throw away
  % uninformative voxels. In the machine learning literature, this is termed
  % 'feature selection'.  The easiest way to do this is to use an ANOVA which
  % tells you the probability that a given voxel's activity varies
  % significantly between conditions over the course of the experiment. 
  %
  % Note: that this ANOVA method runs completely separately for each voxel,
  % yielding a p value for each.
  
  m_mask_xvalid = 'read_mask';
  if (featSel)
    m_mask_xvalid = 'epis_z_thresh0.05';
    
    % Need to DO feature selection
    disp('++ starting feature selection ANOVA')
    subj = feature_select(subj,'epis_z',...
                          sprintf('regs_sh%d',shiftTRs), ...
                          'runs_xval');
  end
  
  %----------------------------------------------------------------------
  % do cross-validation of phase1
  %
  % Use the Matlab Neural Network toolbox
  class_args.train_funct_name = 'train_L2_RLR'; 
  class_args.test_funct_name = 'test_L2_RLR'; 
  class_args.penalty = PENALTY; %% 'PENALTY' is new argument to script
  
  for iter = 1:nIters
    disp(sprintf('++ cross-validation (iter %d of %d)',iter,nIters));
    
    [subj results] = ...
        cross_validation(subj, ...
                         'epis_z', ...
                         sprintf('regs_sh%d',shiftTRs), ...
                         'runs_xval', ...
                         m_mask_xvalid,...
                         class_args);      
  end
  
  % Combine all the results into 1 file.
%   for i = 1:nIters
%     system(sprintf(['cat %s/%s_%02d.txt >> ' ...
%                     '%s/%s_%02d_combined.txt'], ...
%                    output_dir, subjID, i, ...
%                    output_dir, subjID, nIters));
%   end
%   
%   % Remove results files from individual iterations.
%   system (sprintf('rm -f %s/%s_??.txt', ...
%                   output_dir,subjID));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% save results to file
%

disp(['Saving results to:', output_dir, 'results.mat']);
save([output_dir, 'results.mat'], 'results');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% compute meditation period results and save to file - optional - change as needed
% Files used with python post-processing
%

disp(sprintf('Computing stats...'));
cd(output_dir)

load('results.mat')
time = 0;
block = [];
condition =[];
correctlyClassified=[];
decision_all =[];
confidence_all =[];
decision =[];
confidence =[];

condition = [condition results.iterations(2).perfmet.desireds];
correctlyClassified = [correctlyClassified results.iterations(2).perfmet.corrects];
decision_all = [decision_all results.iterations(2).perfmet.guesses];
confidence_all = [confidence_all results.iterations(2).acts];
block = [block ones(1, length(results.iterations(2).perfmet.desireds))];
% remove indices for instructions and reminders
decision = decision_all([7:186 191:250 257:556 561:620]);
confidence = confidence_all(:,[7:186 191:250 257:556 561:620]);

accuracy = results.total_perf;
correctFrequency = [0 0 0 0 0];
frequency = [0 0 0 0 0];

events = [0 0 0 0 0];
events(decision(1)) = 1;

last = decision(1);
lastCondition = condition(1);
frequency = [0 0 0 0 0];
events = [0 0 0 0 0];
events(decision(1)) = 1;


durations = {[], [], [], [], []};
durations{decision(1)} = [1];

activations = {{}, {}, {}, {}, {}};

trialNum = 1;
lastBlock = 1;
lastCondition = condition(1);

for i=2:length(decision)
  frequency(decision(i)) = frequency(decision(i)) + 1;
  if decision(i) ~= last
      events(decision(i)) = events(decision(i)) + 1;
      durations{decision(i)} = [durations{decision(i)} 1];
  else
      durations{decision(i)}(end) = durations{decision(i)}(end) + 1; 
  end
  
  if lastCondition ~= condition(i)
      trialNum = trialNum + 1;
  end
  
  if block(i) ~= lastBlock
      trialNum = 1;
  end
  
  if decision(i) == condition(i)
      correctFrequency(decision(i)) = correctFrequency(decision(i)) + 1;
  end
  
  
  activations{condition(i)} = [activations{condition(i)} confidence(:,i)];
  
  block(i) = block(i) + .01 * trialNum;
  
  last = decision(i);
  lastCondition = condition(i);
  lastBlock = block(i);
end


activationMeans = {[]; []; []; []; []};
activationSDs = {[]; []; []; []; []};

for j=1:5
    conditions = {[], [], [], [], []};
    for i=1:length(activations{j})
        for k=1:5
            conditions{k} = [conditions{k} activations{j}{i}(k)];
        end
    end
    for q=1:5
        activationMeans{j} = [activationMeans{j}, mean(conditions{q})];
        activationSDs{j} = [activationSDs{j}, std(conditions{q})];
    end
end

percent = (frequency / 600) * 100;
accuracy = (correctFrequency/432) * 100;

stats.subjID = subjID;
stats.percent = percent;
stats.events = events;
stats.mean_duration = [mean(durations{1}), mean(durations{2}), mean(durations{3}), mean(durations{4}), mean(durations{5})];
stats.sd_duration = [std(durations{1}), std(durations{2}), std(durations{3}), std(durations{4}), std(durations{5})];
%stats.accuracy = accuracy;
stats.activationMeans = activationMeans;
stats.activationSDs = activationSDs;


save stats_step2.mat stats;

time = linspace(1,length(confidence),length(confidence));
prettyResult.subjID = subjID;
%prettyResult.accuracy = accuracy;
prettyResult.time = time;
prettyResult.block = block;
%prettyResult.condition = condition;
%prettyResult.classCorrect = correctlyClassified;
prettyResult.decision = decision;
prettyResult.confidence = confidence;
prettyResult.decision_all = decision_all;
prettyResult.confidence_all = confidence_all;

save pretty_results_step2.mat prettyResult '-v7';

diary off

cd(sprintf('%s/%s/scripts/batch_scripts/', root_dir, label));

clear

