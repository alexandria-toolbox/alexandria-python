# imports
from alexandria.processor.input_processor import *
from alexandria.results.results import *
from alexandria.graphics.graphics import *
from alexandria.interface.graphical_user_interface import GraphicalUserInterface
from alexandria.results.vector_autoregression_results import *
import alexandria.console.console_utilities as cu


def nowcasting_main_code(user_inputs):
    

    #---------------------------------------------------
    # Alexandria header and estimation start
    #--------------------------------------------------- 


    # display Alexandria header
    cu.print_alexandria_header()
    cu.print_start_message()
    
    
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

    # recover parameters specific to nowcasting (interface 2)
    model = ip.now_model    
    iterations = ip.now_iterations
    burnin = ip.now_burnin
    model_credibility = ip.now_model_credibility   
    
    # recover mfbvar parameters
    if model == 1:    
        constant = ip.mfbvar_constant
        trend = ip.mfbvar_trend
        quadratic_trend = ip.mfbvar_quadratic_trend
        decomposition = ip.mfbvar_decomposition
        lags = ip.mfbvar_lags     
        ar_coefficients = ip.mfbvar_ar_coefficients
        pi1 = ip.mfbvar_pi1
        pi2 = ip.mfbvar_pi2
        pi3 = ip.mfbvar_pi3
        pi4 = ip.mfbvar_pi4
        decomposition_file = ip.mfbvar_decomposition_file

    # recover bdfm parameters
    elif model == 2:
        factors = ip.dfm_factors
        loadings_lags = ip.dfm_loadings_lags
        factor_lags = ip.dfm_factor_lags
        residual_lags = ip.dfm_residual_lags
        sigma = ip.dfm_sigma
        omega = ip.dfm_omega
        delta1 = ip.dfm_delta1
        pi1 = ip.dfm_pi1
        pi2 = ip.dfm_pi2
        pi3 = ip.dfm_pi3
        omega1 = ip.dfm_omega1      
        
    # recover midas parameters
    elif model == 3:
        endogenous_lags = ip.midas_endogenous_lags  
        exogenous_lags = ip.midas_exogenous_lags
        polynomial_order = ip.midas_polynomial_order
        representation = ip.midas_representation
        prior_type = ip.midas_prior_type
        omega1 = ip.midas_omega1
        omega2 = ip.midas_omega2
        upsilon1 = ip.midas_upsilon1
        upsilon2 = ip.midas_upsilon2  
    
    # recover application parameters (interface 3)
    forecast = ip.forecast
    forecast_credibility = ip.forecast_credibility
    conditional_forecast = ip.conditional_forecast
    conditional_forecast_credibility = ip.conditional_forecast_credibility
    irf = ip.irf
    irf_credibility = ip.irf_credibility
    fevd = ip.fevd
    fevd_credibility = ip.fevd_credibility
    hd = ip.hd
    hd_credibility = ip.hd_credibility
    forecast_periods = ip.forecast_periods
    conditional_forecast_type = ip.conditional_forecast_type
    forecast_file = ip.forecast_file
    conditional_forecast_file = ip.conditional_forecast_file
    forecast_evaluation = ip.forecast_evaluation
    irf_periods = ip.irf_periods
    structural_identification = ip.structural_identification
    structural_identification_file = ip.structural_identification_file
    
    # recover remaining parameters
    endogenous = ip.now_endogenous
    exogenous = ip.now_exogenous
    dates = ip.now_dates
    forecast_dates = ip.now_forecast_dates
    Y_p = ip.now_Y_p
    Z_p = ip.now_Z_p
    decomposition_table = ip.now_decomposition_table
    condition_table = ip.now_condition_table
    shock_table = ip.now_shock_table
    restriction_table = ip.now_restriction_table

    # initialize timer
    ip.input_timer('start')
    

    #---------------------------------------------------
    # Model creation
    #--------------------------------------------------- 

    
    # Mixed Frequency Bayesian VAR
    if model == 1:    
        from alexandria.nowcasting.mixed_frequency_bayesian_var import MixedFrequencyBayesianVar
        nowcaster = MixedFrequencyBayesianVar(endogenous, exogenous = exogenous, decomposition = decomposition,
                    decomposition_table = decomposition_table, structural_identification = structural_identification,
                    restriction_table = restriction_table, lags = lags, constant = constant, 
                    trend = trend, quadratic_trend = quadratic_trend, ar_coefficients = ar_coefficients,
                    pi1 = pi1, pi2 = pi2, pi3 = pi3, pi4 = pi4, credibility_level = model_credibility,
                    iterations = iterations, burnin = burnin, verbose = progress_bar)

    # Bayesian dynamic factor model
    elif model == 2:
        from alexandria.nowcasting.bayesian_dynamic_factor_model import BayesianDynamicFactorModel
        nowcaster = BayesianDynamicFactorModel(endogenous, factors = factors, loadings_lags = loadings_lags,
                    factor_lags = factor_lags, residual_lags = residual_lags, sigma = sigma,
                    omega = omega, delta1 = delta1, pi1 = pi1, pi2 = pi2, pi3 = pi3, omega1 = omega1,
                    credibility_level = model_credibility, iterations = iterations, 
                    burnin = burnin, verbose = progress_bar)
    
    # Bayesian MIDAS regression
    elif model == 3:
        from alexandria.nowcasting.bayesian_midas_regression import BayesianMidasRegression
        nowcaster = BayesianMidasRegression(endogenous, exogenous, endogenous_lags = endogenous_lags,
                    exogenous_lags = exogenous_lags, representation = representation, prior_type = prior_type, 
                    omega1 = omega1, omega2 = omega2, upsilon1 = upsilon1, upsilon2 = upsilon2,
                    polynomial_order = polynomial_order, credibility_level = model_credibility,
                    iterations = iterations, burnin = burnin, verbose = progress_bar)
        

    #---------------------------------------------------
    # Model estimation
    #---------------------------------------------------
    
    
    # model estimation
    nowcaster.estimate()


    #---------------------------------------------------
    # Model application: in-sample fit and residuals
    #--------------------------------------------------- 
    
    
    # in-sample fit and residuals
    nowcaster.insample_fit()


    #---------------------------------------------------
    # Model application: forecasts
    #---------------------------------------------------          
        

    # estimate forecasts, if selected
    if forecast:
        
        # Mixed Frequency Bayesian VAR
        if model == 1:
            nowcaster.forecast(forecast_periods, forecast_credibility, Z_p)           
            
        # Bayesian dynamic factor model
        elif model == 2:
            nowcaster.forecast(forecast_periods, forecast_credibility)
                
        # Bayesian MIDAS regression
        elif model == 3:
            for j in range(forecast_periods):
                nowcaster.forecast(j+1, forecast_credibility)
                
        # estimate forecast evaluation, if selected                
        if forecast_evaluation:
            nowcaster.forecast_evaluation(Y_p)                 


    #---------------------------------------------------
    # Model application: impulse response function
    #--------------------------------------------------- 


    # estimate impulse response function, if selected
    if irf and (model == 1 or model == 2):
        nowcaster.impulse_response_function(irf_periods, irf_credibility)
        

    #------------------------------------------------------------
    # Model application: forecast error variance decomposition
    #------------------------------------------------------------
    
    
    # estimate forecast error variance decompositionn, if selected
    if fevd and (model == 1 or model == 2):
        nowcaster.forecast_error_variance_decomposition(irf_periods, fevd_credibility)


    #------------------------------------------------------------
    # Model application: historical decomposition
    #------------------------------------------------------------


    # estimate historical decomposition, if selected
    if hd and (model == 1 or model == 2):
        nowcaster.historical_decomposition(hd_credibility)

    
    #------------------------------------------------------------
    # Model application: conditional forecasts
    #------------------------------------------------------------


    # estimate conditional forecasts, if selected
    if model == 1 and conditional_forecast and conditional_forecast_type == 1:
        nowcaster.conditional_forecast(forecast_periods, conditional_forecast_credibility, condition_table, [], 1, Z_p)
    elif model == 1 and conditional_forecast and conditional_forecast_type == 2:  
        nowcaster.conditional_forecast(forecast_periods, conditional_forecast_credibility, condition_table, shock_table, 2, Z_p) 

    
    #-------------------------------------------------------------
    # Model processor: prepare elements for results and graphics
    #-------------------------------------------------------------        
          
    
    # print estimation completion
    cu.print_completion_message(progress_bar)

    # end estimation timer
    ip.input_timer('end')
    
    # make information dictionary for result class
    ip.make_results_information()
    results_information = ip.results_information
    
    # make information dictionary for graphics class
    ip.make_graphics_information()
    graphics_information = ip.graphics_information
    

    #---------------------------------------------------
    # Model results: create, display and save
    #---------------------------------------------------               
            
           
    # recover path to result folder
    results_path = join(project_path, 'results')

    # initialize results class
    res = Results(nowcaster, results_information)

    # create and save input summary if relevant
    if save_results:
        res.make_input_summary()
        res.save_input_summary(results_path)

    # create, show and save estimation summary if relevant
    res.make_estimation_summary()
    res.show_estimation_summary()
    if save_results:
        res.save_estimation_summary(results_path)

    # create and save application summary if relevant
    if save_results:
        res.make_application_summary()
        res.save_application_summary(results_path)


    #---------------------------------------------------
    # Model graphics: generate and save
    #---------------------------------------------------


    # if graphic selection is selected
    if create_graphics:
        
        # recover path to result folder
        graphics_path = join(project_path, 'graphics')

        # initialize graphics class
        grp = Graphics(nowcaster, graphics_information, graphics_path, True)

        # run graphics for all applications inturn
        grp.insample_fit_graphics(False, True)
        grp.forecast_graphics(False, True)
        grp.conditional_forecast_graphics(False, True)
        grp.irf_graphics(False, True)
        grp.fevd_graphics(False, True)
        grp.hd_graphics(False, True)
        
        # display graphics
        gui = GraphicalUserInterface(view_graphics = True)   


    #---------------------------------------------------
    # Final model return
    #--------------------------------------------------- 
    

    return nowcaster







