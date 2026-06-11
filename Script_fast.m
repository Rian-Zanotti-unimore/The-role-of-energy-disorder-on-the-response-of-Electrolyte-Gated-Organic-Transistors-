%% fit and plot

clc; clear; close all;

%% fitting function call
fit_function = 'FIT_function_name'; 

if exist(fit_function, 'file') ~= 2
    error('fitting function does not exist: %s', fit_function);
end

%% data upload
data_dir = 'folder_name';

file_info = {
    struct('filename', 'file_name.dat', 'Vfixed', -0.1, 'mode', 'Vgs') %transfer
    struct('filename', 'file_name.dat', 'Vfixed', 0.4, 'mode', 'Vds') %output
    };

% parameters initialization and bond: the number and the order of the
% parameters depend on the fitting function 
params0 = [0.1, 5e-2, 1.2, 0.8]; %starting values
lb = [0.05, 1e-4, 0.8, 0.8]; % lower bond
ub = [0.15, 1e-1, 1.6, 0.8]; % upper bond

param_names = {'sigma','gml','Vt','eps'};   % depend on the fitting function 

% select the weighting function
w1 = 1;  % log
w2 = 0;  % lin

%% preprocess 
nfiles = numel(file_info);
all_data = cell(nfiles,1);

for i = 1:nfiles
    data = readtable(fullfile(data_dir, file_info{i}.filename));

    if strcmp(file_info{i}.mode,'Vgs')
        x_data = data.Vgs(2:450); %select just one way
        y_data = data.Ids(2:450);
    else
        x_data = data.Vds(2:51); % exclude Vds = 0
        y_data = data.Ids(2:51); % exclude Vds = 0
    end

    x_data = x_data(:);
    y_data = y_data(:);

    % derivative of the data to compare transconductance
    dx = mean(diff(x_data));
    dy_data  = gradient(y_data, dx);
    d2y_data = gradient(dy_data, dx);

    all_data{i} = struct('x', x_data, 'y', y_data, ...
                         'dy', dy_data, 'd2y', d2y_data, ...
                         'Vfixed', file_info{i}.Vfixed, ...
                         'mode', file_info{i}.mode);
end

%% fitting algorithm 
disp('=== Inizio ottimizzazione globale (GlobalSearch) ===');

opts = optimoptions('fmincon', ...
    'Display', 'iter', ...
    'Algorithm', 'sqp', ... 
    'HessianApproximation','bfgs',...
    'MaxFunctionEvaluations', 5e5, ... 
    'MaxIterations', 5e4, ... 
    'TolFun', 1e-15, ... 
    'TolX', 1e-16); 

problem = createOptimProblem('fmincon', ...
    'objective', @(p) total_obj(p, all_data, w1, w2), ...
    'x0', params0, ...
    'lb', lb, ...
    'ub', ub, ...
    'options', opts);

gs = GlobalSearch('Display', 'iter', 'NumTrialPoints', 500);
[xgs, fgs, exitflag_gs, output_gs, solutions] = run(gs, problem);
disp('=== Fine ottimizzazione globale ===');


problem.x0 = xgs;
[x_final, fval] = fmincon(problem);

xgs = x_final;

%% jacobian and errors

disp('=== Calcolo Jacobiana numerica per stima errori ===')

res_fun = @(p) residual_vector(p, all_data);

J = finite_jacobian(res_fun, xgs);

res = res_fun(xgs);

N = length(res);        
p = length(xgs);        

chi2 = sum(res.^2);
s2 = chi2/(N - p);

Cov = s2 * pinv(J' * J);

param_err = 1.96.*sqrt(diag(Cov));

%% plot section
Vgs_curves = {};
Vds_curves = {};

for i = 1:nfiles
    if strcmp(all_data{i}.mode,'Vgs')
        Vgs_curves{end+1} = all_data{i};
    else
        Vds_curves{end+1} = all_data{i};
    end
end

nVgs = numel(Vgs_curves);
nVds = numel(Vds_curves);

colors_manual = [1 0 0; 0 0 1; 0 1 0; 0 0 0; 0.7 0.45 0.7; ...
                 0.1 0.2 0.3; 0.2 0.2 0.2; 0.1 0.1 0.1; ...
                 0.9 0.09 0.9; 0.4 0.4 0.05; 0.01 0.02 0.4; ...
                 0.4 0.5 0.7];


if ~isempty(Vgs_curves) || ~isempty(Vds_curves)
    colorsVgs = colors_manual(1:nVgs,:);
    colorsVds = colors_manual(1:nVds,:);

    %% Output curves (lin e loglin)
    if nVds > 0
        figure('Position',[100 100 1400 700]);
        for i = 1:nVds
            x = Vds_curves{i}.x;
            y = Vds_curves{i}.y;
            y_fit = feval(fit_function, x, xgs, Vds_curves{i}.Vfixed, 'Vds');

            % Lin
            ax1 = subplot(1,2,1); hold(ax1,'on');
            plot(ax1, x, 1e3*y, 'o','MarkerFaceColor','w','LineWidth',0.5,'Color',colorsVds(i,:));
            plot(ax1, x, 1e3*y_fit, '-','LineWidth',3,'Color',colorsVds(i,:));
            ylabel(ax1,'I_{DS} (mA)'); xlabel(ax1,'V_{DS} (V)');
            grid(ax1,'on'); box(ax1,'on'); axis square;

            % Log-lin
            ax2 = subplot(1,2,2); hold(ax2,'on');
            plot(ax2, x, abs(y), 'o','MarkerFaceColor','w','LineWidth',0.5,'Color',colorsVds(i,:));
            plot(ax2, x, abs(y_fit), '-','LineWidth',3,'Color',colorsVds(i,:));
            set(ax2,'YScale','log');
            ylabel(ax2,'|I_{DS}| (A)'); xlabel(ax2,'V_{DS} (V)');
            grid(ax2,'on'); box(ax2,'on'); axis square;

            set([ax1,ax2],'LineWidth',3,'FontSize',14);
        end
    end

    %% Transfer curves (lin e loglin)
    if nVgs > 0
        figure('Position',[100 100 1400 700]);
        for i = 1:nVgs
            x = Vgs_curves{i}.x;
            y = Vgs_curves{i}.y;
            y_fit = feval(fit_function, x, xgs, Vgs_curves{i}.Vfixed, 'Vgs');

            % Lin
            ax3 = subplot(1,2,1); hold(ax3,'on');
            plot(ax3, x, 1e3*y, 'o','MarkerFaceColor','w','LineWidth',0.5,'Color',colorsVgs(i,:));
            plot(ax3, x, 1e3*y_fit, '-','LineWidth',3,'Color',colorsVgs(i,:));
            ylabel(ax3,'I_{DS} (mA)'); xlabel(ax3,'V_{GS} (V)');
            grid(ax3,'on'); box(ax3,'on'); axis square;

            % Log-lin
            ax4 = subplot(1,2,2); hold(ax4,'on');
            plot(ax4, x, abs(y), 'o','MarkerFaceColor','w','LineWidth',0.5,'Color',colorsVgs(i,:));
            plot(ax4, x, abs(y_fit), '-','LineWidth',3,'Color',colorsVgs(i,:));
            set(ax4,'YScale','log');
            ylabel(ax4,'|I_{DS}| (A)'); xlabel(ax4,'V_{GS} (V)');
            grid(ax4,'on'); box(ax4,'on'); axis square;

            set([ax3,ax4],'LineWidth',3,'FontSize',14);
        end
    end

    %% Transconductance and second derivative
    if nVgs > 0
        figure('Position',[100 100 1400 700]);
        for i = 1:nVgs
            x = Vgs_curves{i}.x;
            y_fit = feval(fit_function, x, xgs, Vgs_curves{i}.Vfixed, 'Vgs');
            dx = mean(diff(x));
            dy_fit  = smoothdata(gradient(y_fit, dx), 'sgolay', 21, 'Degree', 2);
            d2y_fit = smoothdata(gradient(dy_fit, dx), 'sgolay', 21, 'Degree', 2);

            dy = smoothdata(Vgs_curves{i}.dy, 'sgolay', 21, 'Degree', 2);
            d2y = smoothdata(Vgs_curves{i}.d2y, 'sgolay', 21, 'Degree', 2);

            % transconductance
            ax5 = subplot(1,2,1); hold(ax5,'on');
            plot(ax5, x, 1e3*dy, 'o','MarkerFaceColor','w','LineWidth',0.5,'Color',colorsVgs(i,:));
            plot(ax5, x, 1e3*dy_fit, '-','LineWidth',3,'Color',colorsVgs(i,:));
            ylabel(ax5,'g_m (mS)'); xlabel(ax5,'V_{GS} (V)');
            grid(ax5,'on'); box(ax5,'on'); axis square;

            % second derivative
            ax6 = subplot(1,2,2); hold(ax6,'on');
            plot(ax6, x, d2y, 'o','MarkerFaceColor','w','LineWidth',0.5,'Color',colorsVgs(i,:));
            plot(ax6, x, d2y_fit, '-','LineWidth',3,'Color',colorsVgs(i,:));
            ylabel(ax6,'d²I/dV_{DS}² (S/V)'); xlabel(ax6,'V_{GS} (V)');
            grid(ax6,'on'); box(ax6,'on'); axis square;

            set([ax5,ax6],'LineWidth',3,'FontSize',14);
        end
    end
end

%% see the contribution to the fit
fprintf('objective function composition:');

tot_lin = 0; tot_log = 0;
for i = 1:nfiles
    x  = all_data{i}.x;
    y  = all_data{i}.y;
    y_fit = feval(fit_function, x, xgs, all_data{i}.Vfixed, all_data{i}.mode);

    log_y_fit  = log10(abs(y_fit));
    log_y_data = log10(abs(y));

    obj_lin = sum((y_fit - y).^2);
    obj_log = sum((log_y_fit - log_y_data).^2);

    tot_lin = tot_lin + obj_lin;
    tot_log = tot_log + obj_log;

    fprintf('\n--- Curve %d ---\n', i);
    fprintf('Termine           Value (raw)     weighted value \n');
    fprintf('I lin     (w2):   %.3e       %.3e\n', obj_lin, w2*obj_lin);
    fprintf('log(I)    (w1):   %.3e       %.3e\n', obj_log, w1*obj_log);
end

obj_tot = w1*tot_log + w2*tot_lin;
fprintf('SUM');
fprintf(': %.4e\n', obj_tot);

%% table
param_names = {'σ', 'gml','Vt','ε'};
f = uifigure('Name','Parameters','Position',[300 300 400 200]);
uitable(f, ...
    'Data', [param_names(:), string(xgs(:))], ...
    'ColumnName', {'Parameter','Value'}, ...
    'Units','normalized', 'Position',[0 0 1 1]);
fprintf('parameters');
for i = 1:length(param_names)
    fprintf('%s: %.4e\n', param_names{i}, xgs(i));
end

%% cost function
function f = total_obj(p, all_data, w1, w2)
    fit_function = evalin('base','fit_function');
    f = 0;
    for k = 1:numel(all_data)
        x  = all_data{k}.x;
        y  = all_data{k}.y;
        dy = all_data{k}.dy;
        d2y = all_data{k}.d2y;

        y_fit = feval(fit_function, x, p, all_data{k}.Vfixed, all_data{k}.mode);
        dx = mean(diff(x));
        dy_fit  = gradient(y_fit, dx);
        d2y_fit = gradient(dy_fit, dx);

        log_y_fit  = log10(abs(y_fit));
        log_y_data = log10(abs(y));

        obj_lin = sum((y_fit - y).^2);
        obj_log = sum((log_y_fit - log_y_data).^2);
        obj_d1  = sum((dy_fit - dy).^2);
        obj_d2  = sum((d2y_fit - d2y).^2);

        f = f + w1*(obj_log + 0*obj_d1) + w2*(obj_lin + 0*obj_d2);
    end
end

%% save the data
semiconductor_name = data_dir;

params = xgs(:).';      
errors = param_err(:).'; 

for k = 1:nfiles
    x  = all_data{k}.x;
    y  = all_data{k}.y;
    y_fit = feval(fit_function, x, xgs, ...
                  all_data{k}.Vfixed, all_data{k}.mode);

 
    if strcmp(all_data{k}.mode,'Vgs')
        fixed_label = 'VDS';
    else
        fixed_label = 'VGS';
    end

    fixed_value = all_data{k}.Vfixed;
    vstr = strrep(sprintf('%.3f', fixed_value), '.', 'p');

    fname = sprintf('file_name_%s_%s_%s.dat', ...
                    semiconductor_name, fixed_label, vstr); % put the name you prefer

    
    nrows = numel(x);
    param_block      = repmat(params, nrows, 1);
    param_err_block  = repmat(errors, nrows, 1);  
    outmat = [x(:), y(:), y_fit(:), param_block, param_err_block];

    % --- Header ---
    header_cols = [{'x','y_data','y_fit'}, param_names, ...
                   strcat(param_names,'_err')];
    header = strjoin(header_cols, '\t');

    fid = fopen(fname,'w');
    if fid == -1
        warning('Impossibile aprire %s per scrittura.', fname);
        continue;
    end

    fprintf(fid, '%s\n', header);

    fmt = ['%.6e\t%.6e\t%.6e' repmat('\t%.6e',1,2*numel(params)) '\n'];
    fprintf(fid, fmt, outmat.');

    fclose(fid);

    fprintf('Saved: %s\n', fname);
end

%% residuals
function r = residual_vector(p, all_data)

fit_function = evalin('base','fit_function');

r = [];

for k = 1:numel(all_data)

    x  = all_data{k}.x;
    y  = all_data{k}.y;

    y_fit = feval(fit_function, x, p, ...
        all_data{k}.Vfixed, all_data{k}.mode);

    r = [r; y_fit - y];

end

end

%% jacobian
function J = finite_jacobian(fun, x)

fx = fun(x);
n = length(fx);
m = length(x);

J = zeros(n,m);

h = 1e-6;

for i = 1:m

    x1 = x;
    x1(i) = x1(i) + h;

    f1 = fun(x1);

    J(:,i) = (f1 - fx)/h;

end

end

Corr = Cov ./ (sqrt(diag(Cov)) * sqrt(diag(Cov))');