# imports
from alexandria.processor.input_processor import *
from alexandria.results.regression_results import *
from alexandria.graphics.regression_graphics import *
from alexandria.interface.graphical_user_interface import GraphicalUserInterface


def linear_regression_main_code(user_inputs):


    #---------------------------------------------------
    # Result Initialization and Alexandria header
    #--------------------------------------------------- 


    # initiate regression results
    res = RegressionResults()


    #---------------------------------------------------
    # User input processing
    #--------------------------------------------------- 


    # recover user inputs from input processor
    ip = InputProcessor(user_inputs)
    ip.process_input()
    
    # recover specification parameters (interface 1)
    project_path = ip.project_path
    progress_bar = ip.progress_bar
    create_graphics = ip.create_graphics
    save_results = ip.save_results
    
    # recover parameters specific to linear regression (interface 2)
    regression_type = ip.regression_type
    iterations = ip.iterations
    burnin = ip.burnin
    model_credibility = ip.model_credibility
    b = ip.b
    V = ip.V
    alpha = ip.alpha
    delta = ip.delta
    g = ip.g
    Q = ip.Q
    tau = ip.tau
    thinning = ip.thinning
    thinning_frequency = ip.thinning_frequency
    q = ip.q
    p = ip.p
    H = ip.H
    constant = ip.constant
    b_constant = ip.b_constant
    V_constant = ip.V_constant
    trend = ip.trend
    b_trend = ip.b_trend
    V_trend = ip.V_trend
    quadratic_trend = ip.quadratic_trend
    b_quadratic_trend = ip.b_quadratic_trend
    V_quadratic_trend = ip.V_quadratic_trend
    insample_fit = ip.insample_fit
    marginal_likelihood = ip.marginal_likelihood
    hyperparameter_optimization = ip.hyperparameter_optimization
    optimization_type = ip.optimization_type    
    
    # recover application parameters (interface 3)
    forecast = ip.forecast
    forecast_credibility = ip.forecast_credibility
    forecast_evaluation = ip.forecast_evaluation
    
    # recover remaining parameters
    endogenous = ip.endogenous
    exogenous = ip.exogenous
    Z = ip.Z
    y_p = ip.y_p
    X_p = ip.X_p
    Z_p = ip.Z_p


    #---------------------------------------------------
    # Result file creation
    #---------------------------------------------------  
    
    
    # initiate regression results
    res.create_result_file(project_path, save_results)


    #---------------------------------------------------
    # Model creation
    #--------------------------------------------------- 
    
    
    # maximum likelihood regression
    if regression_type == 1:
        from alexandria.linear_regression.maximum_likelihood_regression import MaximumLikelihoodRegression
        lr = MaximumLikelihoodRegression(endogenous, exogenous, constant = constant, trend = trend, \
             quadratic_trend = quadratic_trend, credibility_level = model_credibility, verbose = progress_bar)

    # simple Bayesian regression
    elif regression_type == 2:
        from alexandria.linear_regression.simple_bayesian_regression import SimpleBayesianRegression
        lr = SimpleBayesianRegression(endogenous, exogenous, constant = constant, trend = trend, \
             quadratic_trend = quadratic_trend, b_exogenous = b, V_exogenous = V, b_constant = b_constant, \
             V_constant = V_constant, b_trend = b_trend, V_trend = V_trend, b_quadratic_trend = b_quadratic_trend, \
             V_quadratic_trend = V_quadratic_trend, credibility_level = model_credibility, verbose = progress_bar)

    # hierarchical Bayesian regression
    elif regression_type == 3:
        from alexandria.linear_regression.hierarchical_bayesian_regression import HierarchicalBayesianRegression
        lr = HierarchicalBayesianRegression(endogenous, exogenous, constant = constant, trend = trend, \
             quadratic_trend = quadratic_trend, b_exogenous = b, V_exogenous = V, b_constant = b_constant, \
             V_constant = V_constant, b_trend = b_trend, V_trend = V_trend, b_quadratic_trend = b_quadratic_trend, \
             V_quadratic_trend = V_quadratic_trend, alpha = alpha, delta = delta, credibility_level = model_credibility, \
             verbose = progress_bar)

    # independent Bayesian regression
    elif regression_type == 4:
        from alexandria.linear_regression.independent_bayesian_regression import IndependentBayesianRegression
        lr = IndependentBayesianRegression(endogenous, exogenous, constant = constant, trend = trend, \
             quadratic_trend = quadratic_trend, b_exogenous = b, V_exogenous = V, b_constant = b_constant, \
             V_constant = V_constant, b_trend = b_trend, V_trend = V_trend, b_quadratic_trend = b_quadratic_trend, \
             V_quadratic_trend = V_quadratic_trend, alpha = alpha, delta = delta, iterations = iterations, \
             burn = burnin, credibility_level = model_credibility, verbose = progress_bar)

    # heteroscedastic Bayesian regression
    elif regression_type == 5:
        from alexandria.linear_regression.heteroscedastic_bayesian_regression import HeteroscedasticBayesianRegression
        lr = HeteroscedasticBayesianRegression(endogenous, exogenous, heteroscedastic = Z, \
            constant = constant, trend = trend, quadratic_trend = quadratic_trend, b_exogenous = b, \
            V_exogenous = V, b_constant = b_constant, V_constant = V_constant, b_trend = b_trend, \
            V_trend = V_trend, b_quadratic_trend = b_quadratic_trend, V_quadratic_trend = V_quadratic_trend, \
            alpha = alpha, delta = delta, g = g, Q = Q, tau = tau, iterations = iterations, \
            burn = burnin, thinning = thinning, thinning_frequency = thinning_frequency, \
            credibility_level = model_credibility, verbose = progress_bar)

    # autocorrelated Bayesian regression
    elif regression_type == 6:
        from alexandria.linear_regression.autocorrelated_bayesian_regression import AutocorrelatedBayesianRegression
        lr = AutocorrelatedBayesianRegression(endogenous, exogenous, q = q, constant = constant, \
            trend = trend, quadratic_trend = quadratic_trend, b_exogenous = b, V_exogenous = V, \
            b_constant = b_constant, V_constant = V_constant, b_trend = b_trend, \
            V_trend = V_trend, b_quadratic_trend = b_quadratic_trend, V_quadratic_trend = V_quadratic_trend, \
            alpha = alpha, delta = delta, p = p, H = H, iterations = iterations, \
            burn = burnin, credibility_level = model_credibility, verbose = progress_bar)      
            

    #---------------------------------------------------
    # Model optimization
    #---------------------------------------------------     
    
    
    # apply if regression is simple Bayesian or hierarchical, and optimization is selected
    if (regression_type == 2 or regression_type == 3) and hyperparameter_optimization:
        lr.optimize_hyperparameters(optimization_type)
    

    #---------------------------------------------------
    # Model estimation
    #---------------------------------------------------
    
    
    # model estimation
    lr.estimate()    
    

    #---------------------------------------------------
    # Model application: in-sample fit and residuals
    #--------------------------------------------------- 
    
    
    # apply if in-sample fit and residuals is selected
    if insample_fit:
        lr.fit_and_residuals()
        
        
    #---------------------------------------------------
    # Model application: marginal likelihood
    #---------------------------------------------------         
    
    
    # apply if marginal likelihood is selected, and model is any Bayesian regression
    if marginal_likelihood and regression_type != 1:
        lr.marginal_likelihood()
        
        
    #---------------------------------------------------
    # Model application: forecasts
    #---------------------------------------------------          
        
        
    # estimate forecasts, if selected
    if forecast:
        if regression_type == 5:
            lr.forecast(X_p, forecast_credibility, Z_p)
        else:   
            lr.forecast(X_p, forecast_credibility)
        # estimate forecast evaluation, if selected
        if forecast_evaluation:
            lr.forecast_evaluation(y_p)
            
            
    #---------------------------------------------------
    # Model results: create, display and save
    #---------------------------------------------------               
            
            
    # create, display and save result summary
    res.result_summary(ip, lr)
    
    # create and save model settings
    res.settings_summary()
    
    # create and save model applications
    res.application_summary()
    
    
    #---------------------------------------------------
    # Model graphics: generate and save
    #---------------------------------------------------


    # if graphic selection is selected
    if create_graphics:
        
        # initiate graphic creation
        rg = RegressionGraphics(ip, lr)
        
        # generate graphics
        rg.make_graphics()
        
        # display graphics
        gui = GraphicalUserInterface(view_graphics = True)
        

    #---------------------------------------------------
    # Final model return
    #--------------------------------------------------- 
    
    
    return lr




