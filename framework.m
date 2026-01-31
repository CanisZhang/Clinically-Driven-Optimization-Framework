%% ===================== SURROGATEOPT + CRN (IMPULSED) =====================
clear; clc;

%% -------- Global Configuration --------
cfg.num_repeats              = 5;           % Outer repeats
cfg.timeACQ                  = 11.6;        % Used in constraints/time calc
cfg.penalty_infeasible       = 1e6;         % Penalty for infeasible/failed eval

% ---- Optimization-stage noise (Cheaper) ----
cfg.num_noise_iterations_optim = 20;

% ---- True-evaluation noise (More accurate) ----
cfg.num_noise_iterations_true  = 50;

% ====== CRN: Fixed randomness during optimization ======
cfg.crn_master_seed = 20260128;             % Keep fixed for reproducibility
rng(cfg.crn_master_seed, 'twister');
cfg.CRNSeeds = randi(2^31-1, 1, cfg.num_noise_iterations_optim);

% ====== True mean evaluation: Independent randomness ======
cfg.num_true_trials  = 30;                  % Trials for true mean verification
cfg.true_master_seed = 13579;

% ====== Feasible initial points sampling ======
cfg.init_feasible_want     = 20;            % Target feasible seeds
cfg.init_feasible_maxdraws = 2000;          % Max random draws

%% Start Parallel Pool
if isempty(gcp('nocreate'))
    parpool;
end

%% Results Container
results.best_params = zeros(cfg.num_repeats, 6); % [Hz1 Hz2 b1 b2 PGSE_delta PGSE_b]
results.fval_crn    = nan(cfg.num_repeats, 1);
results.true_mean   = nan(cfg.num_repeats, 1);
results.true_std    = nan(cfg.num_repeats, 1);
results.true_ci95   = nan(cfg.num_repeats, 2);

%% ===================== Outer Repeats =====================
for r = 1:cfg.num_repeats
    fprintf('\n========== Starting Run %d/%d ==========\n', r, cfg.num_repeats);
    
    % Run Optimization
    [best_params, fval_crn, exitflag, output] = optimize_IMPULSED_surrogateopt_CRN(cfg);
    
    results.best_params(r,:) = best_params(:).';
    results.fval_crn(r)      = fval_crn;
    
    fprintf('Run %d finished: fval=%.6f, exitflag=%d, evals=%d\n', ...
        r, fval_crn, exitflag, output.funccount);

    % Post-optimization: True mean check
    stats = evaluate_true_mean(best_params, cfg);
    results.true_mean(r)   = stats.mean;
    results.true_std(r)    = stats.std;
    results.true_ci95(r,:) = stats.ci95;
    
    fprintf('True Eval: mean=%.6f, std=%.6f, 95%%CI=[%.6f, %.6f]\n', ...
        stats.mean, stats.std, stats.ci95(1), stats.ci95(2));
end

disp('========== ALL DONE ==========');
disp(results);


%% ===================== LOCAL FUNCTIONS =====================

function [best_params_phys, fval, exitflag, output] = optimize_IMPULSED_surrogateopt_CRN(cfg)
    % Decision variables: [OGSEHz1, OGSEHz2, alpha1, alpha2, PGSE_delta, PGSE_b]
    % alpha1/alpha2 in [0,1] map to feasible b1/b2 automatically.
    
    OGSEHz1_range     = [17, 40];
    OGSEHz2_range     = [30, 65];
    alpha_range       = [0, 1];
    PGSE_delta_range  = [30, 90];
    PGSE_b_range      = [300, 1000];

    lb = [OGSEHz1_range(1), OGSEHz2_range(1), alpha_range(1), alpha_range(1), PGSE_delta_range(1), PGSE_b_range(1)];
    ub = [OGSEHz1_range(2), OGSEHz2_range(2), alpha_range(2), alpha_range(2), PGSE_delta_range(2), PGSE_b_range(2)];

    % Helper for batch size
    pool = gcp('nocreate');
    if isempty(pool)
        parpool; pool = gcp('nocreate');
    end
    batchN = max(1, pool.NumWorkers);

    % Find feasible initial points
    initPts = generate_initial_feasible_points(cfg, lb, ub);
    if ~isempty(initPts)
        fprintf('Found %d feasible InitialPoints.\n', size(initPts,1));
    else
        fprintf('No feasible InitialPoints found (using random start).\n');
    end

    % Surrogateopt options
    options = optimoptions('surrogateopt', ...
        'Display', 'iter', ...
        'UseParallel', true, ...
        'PlotFcn', 'surrogateoptplot', ...
        'MaxFunctionEvaluations', 1500, ...
        'MinSurrogatePoints', 20, ...
        'BatchUpdateInterval', batchN, ...
        'ConstraintTolerance', 1e-3);

    if ~isempty(initPts)
        options = optimoptions(options, 'InitialPoints', initPts);
    end

    % Objective function
    fun = @(x) objconstr_IMPULSED_CRN(x, cfg);
    
    [xbest, fval, exitflag, output] = surrogateopt(fun, lb, ub, options);
    
    % Convert optimal x back to physical parameters
    best_params_phys = x_to_physical_params(xbest, cfg);
end

% -----------------------------------------------------------
function out = objconstr_IMPULSED_CRN(x, cfg)
    % 1. Check constraints
    out.Ineq = nonlinear_constraints_ineq_x(x, cfg);
    
    % 2. Quick penalty for infeasible points
    if any(~isfinite(out.Ineq)) || any(out.Ineq > 0)
        v = max(out.Ineq, 0);
        out.Fval = cfg.penalty_infeasible + 1e3 * sum(v.^2);
        return;
    end

    % 3. Run CRN evaluation if feasible
    phys = x_to_physical_params(x, cfg);
    out.Fval = run_IMPULSEDsimulation_withSeeds_phys(phys, cfg, cfg.CRNSeeds, cfg.num_noise_iterations_optim);
end

% -----------------------------------------------------------
function c = nonlinear_constraints_ineq_x(x, cfg)
    % x = [Hz1 Hz2 alpha1 alpha2 PGSE_delta PGSE_b]
    Hz1        = x(1);
    Hz2        = x(2);
    PGSE_delta = x(5);

    % Base constraint: Hz2 >= Hz1 + 2
    c1 = Hz1 - Hz2 + 2;

    % Time feasibility
    TE = minTE(PGSE_delta, Hz1, Hz2);
    if ~isfinite(TE) || TE <= 0
        c = [Inf; Inf; Inf; Inf; Inf; Inf]; return;
    end
    
    duringTime = TE/2 - cfg.timeACQ;
    epsT = 1e-6;
    c2 = -duringTime + epsT; 

    if ~isfinite(duringTime) || duringTime <= 0
        c = [c1; Inf; Inf; c2; Inf; Inf]; return;
    end

    % Cycle counts (N) constraints
    OGSEdelta1 = 1000/Hz1;
    N1 = floor(duringTime / OGSEdelta1);
    
    OGSEdelta2 = 1000/Hz2;
    N2 = floor(duringTime / OGSEdelta2);
    
    c3 = -N1 + 1; % N1 >= 1
    c4 = -N2 + 1; % N2 >= 1

    if N1 < 1 || N2 < 1
        c = [c1; Inf; Inf; c2; c3; c4]; return;
    end

    % B-value feasibility
    maxb1 = OGSEbmax(Hz1, N1);
    maxb2 = OGSEbmax(Hz2, N2);
    
    if ~isfinite(maxb1) || ~isfinite(maxb2)
        c = [c1; Inf; Inf; c2; c3; c4]; return;
    end

    c5 = 300 - maxb1; % maxb1 >= 300
    c6 = 300 - maxb2; % maxb2 >= 300

    c = [c1; c5; c6; c2; c3; c4];
end

% -----------------------------------------------------------
function phys = x_to_physical_params(x, cfg)
    % Convert optimized variables to physical: [Hz1 Hz2 b1 b2 PGSE_delta PGSE_b]
    Hz1        = x(1);
    Hz2        = x(2);
    a1         = x(3);
    a2         = x(4);
    PGSE_delta = x(5);
    PGSE_b     = x(6);

    TE = minTE(PGSE_delta, Hz1, Hz2);
    duringTime = TE/2 - cfg.timeACQ;
    
    OGSEdelta1 = 1000/Hz1; N1 = floor(duringTime / OGSEdelta1);
    OGSEdelta2 = 1000/Hz2; N2 = floor(duringTime / OGSEdelta2);
    
    maxb1 = OGSEbmax(Hz1, N1);
    maxb2 = OGSEbmax(Hz2, N2);
    
    b1_up = min(800, maxb1);
    b2_up = min(800, maxb2);

    if ~isfinite(b1_up) || ~isfinite(b2_up) || b1_up < 300 || b2_up < 300
        phys = [Hz1, Hz2, NaN, NaN, PGSE_delta, PGSE_b];
        return;
    end

    b1 = 300 + a1 * (b1_up - 300);
    b2 = 300 + a2 * (b2_up - 300);
    phys = [Hz1, Hz2, b1, b2, PGSE_delta, PGSE_b];
end

% -----------------------------------------------------------
function initPts = generate_initial_feasible_points(cfg, lb, ub)
    % Monte Carlo sampling for constraint-satisfying starting points
    nWant   = cfg.init_feasible_want;
    maxDraw = cfg.init_feasible_maxdraws;
    nvars   = numel(lb);
    initPts = zeros(nWant, nvars);
    cnt     = 0;

    for k = 1:maxDraw
        x = lb + rand(1, nvars) .* (ub - lb);
        c = nonlinear_constraints_ineq_x(x, cfg);
        if all(isfinite(c)) && all(c <= 0)
            cnt = cnt + 1;
            initPts(cnt,:) = x;
            if cnt >= nWant, break; end
        end
    end
    initPts = initPts(1:cnt,:);
    if isempty(initPts), initPts = []; end
end

% -----------------------------------------------------------
function err = run_IMPULSEDsimulation_withSeeds_phys(phys, cfg, seeds, num_noise_iterations)
    % Core simulation function using Common Random Numbers (CRN)
    
    % Fixed Params
    SNR0  = 6; TE0 = 88; T2 = 40; trise = 1.54;
    
    errs = nan(1, num_noise_iterations);
    fail_count = 0;

    OGSEHz1    = phys(1);
    OGSEHz2    = phys(2);
    OGSEb1     = phys(3) ./ 1000;
    OGSEb2     = phys(4) ./ 1000;
    PGSE_delta = phys(5);
    PGSE_b     = phys(6) ./ 1000;

    if any(~isfinite(phys)), err = cfg.penalty_infeasible; return; end

    % Timing calculations
    TE = minTE(PGSE_delta, OGSEHz1, OGSEHz2);
    duringTime = TE/2 - cfg.timeACQ;
    
    if ~isfinite(TE) || TE <= 0 || ~isfinite(duringTime) || duringTime <= 0
        err = cfg.penalty_infeasible; return;
    end

    OGSEdelta1 = 1000/OGSEHz1;
    OGSEN1 = floor(duringTime / OGSEdelta1);
    if OGSEN1 <= 0, err = cfg.penalty_infeasible; return; end
    OGSEdelta1 = OGSEN1 * OGSEdelta1;

    OGSEdelta2 = 1000/OGSEHz2;
    OGSEN2 = floor(duringTime / OGSEdelta2);
    if OGSEN2 <= 0, err = cfg.penalty_infeasible; return; end
    OGSEdelta2 = OGSEN2 * OGSEdelta2;

    % Build Pulse Sequence
    pulse_tcos1 = mati.DiffusionPulseSequence(4, 'TE', TE, 'delta', OGSEdelta1, ...
        'Delta', OGSEdelta1 + 6, 'b', 0:OGSEb1/3:OGSEb1, 'n', OGSEN1, 'shape', "tcos", 'trise', trise);
    
    pulse_tcos2 = mati.DiffusionPulseSequence(4, 'TE', TE, 'delta', OGSEdelta2, ...
        'Delta', OGSEdelta2 + 6, 'b', 0:OGSEb2/3:OGSEb2, 'n', OGSEN2, 'shape', "tcos", 'trise', trise);
    
    pulse_tpgse = mati.DiffusionPulseSequence(4, 'TE', TE, 'delta', 12, ...
        'Delta', PGSE_delta, 'b', 0:PGSE_b/3:PGSE_b, 'shape', "tpgse", 'trise', trise);

    pulse = mati.PulseSequence.cat(pulse_tpgse, pulse_tcos1, pulse_tcos2);

    % Model Setup
    nmodel = 1;
    switch nmodel
        case 1
            structure.modelName = 'IMPULSED_vin_d_Dex'; structure.Din = 2;
            vin = [0.4 0.2 0.6]; d = [14 11 19]; Dex = [2.4 1.6 1.0];
            [vin, d, Dex] = meshgrid(vin, d, Dex);
            parms_sim = {vin, d, Dex};
        case 2
            structure.modelName = 'IMPULSED_vin_d_Dex_Din';
            vin = 0.6; d = 10:15; Dex = [1.56 3];
            [vin, d, Dex] = meshgrid(vin, d, Dex);
            parms_sim = {vin, d, Dex};
        case 3
            structure.modelName = 'IMPULSED_vin_d_Dex_Din_betaex';
            vin = [0.5 0.7]; d = [8 16 24]; Dex = [1.56 3]; Din = 1.56;
            [vin, d, Dex, Din] = ndgrid(vin, d, Dex, Din);
            parms_sim = {vin, d, Dex, Din};
    end
    impulsed = mati.IMPULSED(structure, pulse);
    signal_sim_noiseless = impulsed.FcnSignal(parms_sim, impulsed);

    % Fitter Setup
    fitopts.flag.denoise = 'y'; fitopts.flag.getADC = 'y'; fitopts.flag.multistart = 'y';
    fitopts.flag.useGPU = 'n'; fitopts.fittingMethod = 'nllsq'; fitopts.flag.parfor = 'n';
    fitopts.NumStarts = 1;
    fitpars = mati.FitPars(impulsed, fitopts);
    
    warning off;
    weight_TE = exp(-(TE - TE0)/T2);

    % Noise Loop
    for iter = 1:num_noise_iterations
        rng(seeds(iter), 'twister');
        
        base0 = mean(signal_sim_noiseless(1,:));
        if ~isfinite(base0) || base0 <= 0, fail_count = fail_count + 1; continue; end
        
        signal_sim = signal_sim_noiseless;
        sigma_vec = nan(size(signal_sim_noiseless,1),1);

        for i = 1:size(signal_sim_noiseless,1)
            mu_i = mean(signal_sim_noiseless(i,:));
            if ~isfinite(mu_i) || mu_i <= 0, continue; end
            
            weight_b_D   = mu_i / base0;
            weight_sqrtN = 4;
            SNR = SNR0 * weight_b_D * weight_TE * weight_sqrtN;
            
            if ~isfinite(SNR) || SNR <= 0, continue; end
            
            sigma_i = mu_i / SNR;
            sigma_vec(i) = sigma_i;
            signal_sim(i,:) = mati.Physics.AddRicianNoise(signal_sim_noiseless(i,:)', sigma_i)';
        end

        if any(~isfinite(signal_sim(:))) || all(~isfinite(sigma_vec))
            fail_count = fail_count + 1; continue;
        end

        sigma_for_data = mean(sigma_vec(isfinite(sigma_vec)));
        if ~isfinite(sigma_for_data) || sigma_for_data <= 0, fail_count = fail_count + 1; continue; end

        [Npulse, Nparms] = size(signal_sim);
        data = mati.ImageData(reshape(signal_sim', [Nparms, 1, 1, Npulse]), sigma_for_data);

        try
            fitout = fitpars.Fit(data);
            isFail = any(~isfinite(fitout.d))   || any(fitout.d   <= 0) || ...
                     any(~isfinite(fitout.vin)) || any(~isfinite(fitout.Dex));
            if isFail
                fail_count = fail_count + 1; continue;
            end
            
            fitcellularity = fitout.vin ./ fitout.d;
            cellularity    = vin ./ d;
            
            % Weighted SSE
            wd = 1; wc = 1; wDex = 1; wvin = 1;
            current_error = 0;
            for i2 = 1:size(signal_sim,2)
                current_error = current_error + ...
                    wvin*((fitout.vin(i2) - vin(i2))/vin(i2))^2 + ...
                    wd  *((fitout.d(i2)   - d(i2))  /d(i2))^2 + ...
                    wDex*((fitout.Dex(i2) - Dex(i2))/Dex(i2))^2 + ...
                    wc  *((fitcellularity(i2) - cellularity(i2))/cellularity(i2))^2;
            end
            errs(iter) = current_error;
        catch
            fail_count = fail_count + 1;
        end
    end

    % Result Aggregation (Median + Failure Penalty)
    succ = errs(isfinite(errs));
    if isempty(succ)
        err = cfg.penalty_infeasible;
    else
        base = median(succ);
        fail_rate = fail_count / num_noise_iterations;
        lambda = max(2 * base, 1);
        err = base + lambda * (fail_rate^2);
    end
end

% -----------------------------------------------------------
function stats = evaluate_true_mean(best_params_phys, cfg)
    % Independent validation of the best result
    nT = cfg.num_true_trials;
    f  = nan(1, nT);
    for t = 1:nT
        rng(cfg.true_master_seed + t, 'twister');
        seeds_t = randi(2^31-1, 1, cfg.num_noise_iterations_true);
        f(t) = run_IMPULSEDsimulation_withSeeds_phys(best_params_phys, cfg, seeds_t, cfg.num_noise_iterations_true);
    end
    stats.mean = mean(f);
    stats.std  = std(f, 0);
    se = stats.std / sqrt(nT);
    stats.ci95 = [stats.mean - 1.96*se, stats.mean + 1.96*se];
end

function maxb = OGSEbmax(frequency, N)
    % Calculate max b-value given frequency and N
    OGSEdelta = N * 1000 / frequency;  
    gmax = 80e-5;
    
    pulse_tcos = mati.DiffusionPulseSequence(1, ...
        'TE', 150, ...              % Echo time
        'delta', OGSEdelta, ...     % Small delta
        'Delta', OGSEdelta + 7, ... % Big Delta
        'G', [gmax], ...            % Max gradient strength
        'n', N * ones(1, 1), ...    % Number of cycles
        'shape', "tcos", ...        % Pulse shape
        'gdir', [0 0 1], ...        % Gradient direction
        'trise', 1.54);             % Rise time
        
    maxb = 1000 * pulse_tcos.b(1); 
end

function TE = minTE(PGSE_delta, OGSEHz1, OGSEHz2)
    timeACQ = 11.6;
    OGSEdelta = max(1000/OGSEHz1, 1000/OGSEHz2);
    TE = max(2*ceil(OGSEdelta+timeACQ), PGSE_delta+35);
end
