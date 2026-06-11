function I_DS = FIT_gauss(x, params, Vfixed, mode)
% x: (Vgs o Vds)
% Vfixed: values of the other bias
% mode: 'Vgs' or 'Vds' (transfer or output)

% parameters
% params = [sigma, alpha, gml, Vt, eps, Vt0]

sigma = params(1);
gml = params(2);
Vt = params(3);
eps = params(4);


% constants
nu_gap = eps / sigma;
nu_T = Vt/sigma;

I_DS = zeros(size(x));

for i = 1:length(x)
    if strcmp(mode, 'Vgs')
        Vgs = x(i);
        Vds = Vfixed;
    elseif strcmp(mode, 'Vds')
        Vgs = Vfixed;
        Vds = x(i);
    else
        error('Mode deve essere "Vgs" o "Vds"');
    end

  
    nu_GS = Vgs / sigma;
    nu_DS = Vds / sigma;
    

    %nu_po = min(1,(nu_GS(k)-nu_T)/nu_DS); decomment to use the pinch-off
    A = (nu_GS-nu_T-nu_gap);
    nu_ds = nu_DS;
    zeta = (1/(2*nu_ds))*(A*erf(A/sqrt(2))+sqrt(2/pi)*exp(-0.5*A^2)-((A-nu_ds)*erf((A-nu_ds)/sqrt(2))+sqrt(2/pi)*exp(-0.5*(A-nu_ds)^2)-nu_ds));
    I_DS(i) = Vds*gml*(zeta);


end

% correlation matrix
params_matrix = [sigma; gml; Vt; eps];
correlation_matrix = corrcoef(params_matrix);