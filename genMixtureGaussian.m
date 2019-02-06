% Generate data from a mixture of 2d gaussians randomly
function X = genMixtureGaussian(N, K, prior_mean, prior_var)
    D = size(prior_mean, 2);

    % generate mixture locations from the prior
    locs = mvnrnd(prior_mean, eye(D) * sqrt(prior_var), K);
    % generate the data based on the locs
    X = zeros(N, D);
    
    % Init the indicator c
    c = zeros(N, 1);
    
    for i = 1: N
        c(i) = randsample(1:K, 1);
        
        % draw the observation from the corresponding mixture location
        X(i,:) = mvnrnd(locs(c(i),:), eye(D));
    end
end