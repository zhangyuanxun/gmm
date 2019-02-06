function em(X, K)
    %% --------- THE EM ALGORITHM ---------
    % (1) Initlize the mean mu, covariance sigma, and mixing coefficients pie,
    % and evaluate the initial value of log likelihood.
    [N, D] = size(X);
    format long g;

    % Randomly select K points from dataset X as mu
    mu = zeros(K, D);
    for k = 1: K
        r = ceil(rand(1) * N);
        mu(k,:) = X(r,:);
    end

    % Randomly generate covariance matric sigma
    sigma = zeros(D, D, K);
    for k = 1: K
        sigma(:,:, k) = genCovMatrix(D);
    end

    % Generate the uniform prior pie p(pie_k) ~ uniform
    pie = ones(1, K) / K;

    % compute the initial value of log likelihood
    log_likelihood = calcLikelihood(X, K, pie, mu, sigma);

    likelihoods = [log_likelihood];
    % (2) E-Step: Evaluate the responsibilities using the current parameter 
    % values.
    count = 1;
    iter_number = 5000;
    plotStart = 0;
    conv = 1;
    while (conv > 1e-10)    
        % plot GMM
        plot_gmm(X, mu, sigma, K, likelihoods);
        if (plotStart == 0)
            pause(0.2);
            plotStart = 1;
        end

        Z = zeros(N, K);
        prob = zeros(N, K);
        for n = 1: N
            for k = 1: K
                prob(n, k) = pie(k) *  mvnpdf(X(n,:), mu(k,:), sigma(:,:,k));
            end

            prob(n,:) = prob(n,:) / sum(prob(n,:));
        end

        % compute the NK, which means the number of points for each cluster
        NK = sum(prob, 1);

        % (3) M step. Re-estimate the parameters using the current responsibilities
        % (3.1) Re-estimate the mu
        for k = 1: K
            s = zeros(1, D);
            for n = 1: N
                s = s + prob(n, k) * X(n,:);
            end
            mu(k,:) = s / NK(k);
        end

        % (3.2) Re-estimate the sigma
        for k = 1: K
            s = zeros(D, D);
            for n = 1: N
                s = s + prob(n, k) * ((X(n,:) - mu(k,:))' * (X(n,:) - mu(k,:)));
            end

            sigma(:,:,k) =  s / NK(k);
        end

        % (3.3) Re-estimate the pie
        pie = NK / N;

        % (4) Evaluate the log likelihood.
        new_log_likelihood = calcLikelihood(X, K, pie, mu, sigma);
        likelihoods = [likelihoods new_log_likelihood];

        conv = abs((likelihoods(end) - likelihoods(end-1))) / abs(likelihoods(end - 1));
        count = count + 1;
        
        fprintf("iteration %d; convergence %g. \n", count, conv);
        pause(0.2);
    end
end

function M = genCovMatrix(n)
    Q = orth(randn(n));
    D = diag(abs(randn(n, 1)) + 0.3);
    M = Q*D*Q';
end

function logl = calcLikelihood(X, K, pie, mu, sigma)
    N = size(X, 1);
    logl = 0;
    for n = 1: N
        l = 0;
        for k = 1: K
            l = l + pie(k) * mvnpdf(X(n,:), mu(k,:), sigma(:,:,k));
        end
        logl = logl + log(l);
    end
end

function plot_gmm(X, mu, sigma, K, likelihoods)
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
    t = ['GMM Clutering Demo, K = ', num2str(K)];
    title(t);
    hold on;
    scale = abs(max(X(:,1)) - min(X(:,1))) * 0.2;
    xlim([min(X(:,1)) - scale max(X(:,1)) + scale + 1]);
    
    scale = abs(max(X(:,2)) - min(X(:,2))) * 0.2;
    ylim([min(X(:,2)) - scale max(X(:,2)) + scale + 1]);
    for k = 1: K
        gaussPlot2d(mu(k,:), sigma(:,:,k));
    end
    axis equal;
    hold off;
    
    % likelihood plot
    subplot(1,2,2);
    plot(likelihoods, 'LineWidth', 2);
    t = ['Log Likelihood = ', num2str(likelihoods(end))];
    title(t);
    hold on;
    scale = abs(max(likelihoods) - min(likelihoods)) * 0.3;
    ylim([min(likelihoods)-scale max(likelihoods)+scale+1]);
    
    xlabel("iterations");
    xlim([1 size(likelihoods, 2) * 1.4]);
    hold off;
    drawnow;
    
    % Write GIF
%     frame = getframe(h); 
%     im = frame2im(frame); 
%     [imind,cm] = rgb2ind(im,256); 
%     if size(likelihoods, 2) == 1 
%         imwrite(imind,cm,'gmm.gif', 'gif', 'Loopcount',inf); 
%     else 
%         imwrite(imind,cm,'gmm.gif', 'gif', 'WriteMode','append'); 
%     end 
end
