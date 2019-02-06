clear all;
% load 2D dataset as X
K = 5;
N = 1000;
prior_var = 2500;

X = genMixtureGaussian(N, K, [0, 0], prior_var);
gmm_vb(X, K, prior_var);

%  variational inference
function gmm_vb(X, K, prior_var)
    [N, D] = size(X);
    
    var_mu = mvnrnd(mean(X), eye(D), K);     % variational mean
    var_var = ones(K, 1) * sqrt(prior_var);  % variational variance
    var_phi = ones(N, K) * (1 / K);          % variational phi for cluster indicator
    
    % compute the initial elbo
    elbo = gaussianELBO(X, K, var_mu, var_var, var_phi, prior_var);
    elbos = [elbo];
    
    % until convergence
    conv = 1;
    iter = 0;
    
    % Plot the initial state 
    plot_gmm(X, var_mu, sqrt(var_var), K, elbos);
    pause(3);
        
    while (conv > 1e-10)
        iter = iter + 1;
        
        % for each data point
        for i = 1: N
            % compute the variational posterior on the cluster
            phi = zeros(K, 1);

            for k = 1: K
                phi(k) = X(i,:) * var_mu(k,:)' - (var_var(k) + var_mu(k,:) * var_mu(k,:)') / 2;
            end
            var_phi(i,:) = exp(phi - logsumexp(phi));
        end

        % for each cluster 
        for k = 1 : K
            var_mu(k,:) = (var_phi(:,k)' * X) / ((1 / prior_var) + sum(var_phi(:,k)));
            var_var(k) = 1 / ((1 / prior_var) + sum(var_phi(:,k)));
        end
        
        elbo = gaussianELBO(X, K, var_mu, var_var, var_phi, prior_var);
        elbos = [elbos elbo];
        
        plot_gmm(X, var_mu, sqrt(var_var), K, elbos);
        conv = abs((elbos(end) - elbos(end-1))) / abs(elbos(end - 1));
        fprintf("iteration %d; convergence %g. \n", iter, conv);
        pause(0.5);
    end
end


% compute ELBO
function elbo = gaussianELBO(X, K, var_mu, var_var, var_phi, prior_var)
    
    [N, D] = size(X);
    elbo = 0;
    
    % E[prior] and entropy of mixture locations
    for k= 1: K
        elbo = elbo + log((1 / sqrt(2 * pi * prior_var))) - ...
             (var_var(k) + var_mu(k,:) * var_mu(k,:)') / (2 * prior_var);  % expected log prior over mixture locations
        
        elbo = elbo + (1 / 2) * (log(2 * pi * var_var(k))) + (1 / 2);     % entropy of each variational location posterior 
    end  
    
    % E[prior on z] + E[likelihood] + entropy of q(z)
    for i = 1: N
        elbo = elbo + log(1 / K);
        elbo = elbo + log (1 / sqrt(2 * pi)) - (X(i,:) * X(i,:)')/ 2;
        
        for k = 1: K
            elbo = elbo + var_phi(i, K) * (X(i,:) * var_mu(k,:)');
            elbo = elbo - var_phi(i, K) * (var_var(k) + var_mu(k,:) * var_mu(k,:)') / 2;
            elbo = elbo - var_phi(i, k) * log(var_phi(i, K));
        end
    end
end

function plot_gmm(X, mu, sigma, K, elbos)
    [N, D] = size(X);
    h = figure(1);
    
    % set figure as the center of screen
    pixels = get(0,'ScreenSize');
    figure_width = pixels(3) * 0.7; 
    figure_height = pixels(4) * 0.47; 
    
    point1 = round(pixels(3)/2 - figure_width/2);
    point2 = round(pixels(4)/2 - figure_height/2);
    point3 = figure_width;
    point4 = figure_height;
    set(h, 'Position', [point1 point2 point3 point4]);
    
    % cluster plot
    subplot(1,2,1);
    scatter(X(:,1), X(:,2), 'filled');
    t = ['Variational Bayesian mixture of Gaussians Demo, K = ', num2str(K)];
    title(t);
    hold on;
    xlim([min(X(:,1)) max(X(:,1))]);
    ylim([min(X(:,2)) max(X(:,2))]);
    for k = 1: K
        gaussPlot2d(mu(k,:), eye(D) * sigma(k));
    end
    axis equal;
    hold off;
    
    % likelihood plot
    subplot(1,2,2);
    plot(elbos, 'LineWidth', 2);
    t = ['ELBO = ', num2str(elbos(end))];
    title(t);
    hold on;
    scale = (max(elbos) - min(elbos)) * 0.2;
    ylim([min(elbos)-scale max(elbos)+scale + 1]);
    xlim([1 size(elbos, 2) * 1.4]);
    xlabel("iterations");
    hold off;
    drawnow;
    
    % Write GIF
%     frame = getframe(h); 
%     im = frame2im(frame); 
%     [imind,cm] = rgb2ind(im,256); 
%     if size(elbos, 2) == 1 
%         imwrite(imind,cm,'gmm_vb.gif', 'gif', 'Loopcount',inf); 
%     else 
%         imwrite(imind,cm,'gmm_vb.gif', 'gif', 'WriteMode','append'); 
%     end 
end