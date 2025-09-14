# imports
from alexandria.processor.input_processor import *
from alexandria.results.results import *
from alexandria.graphics.graphics import *
from alexandria.interface.graphical_user_interface import GraphicalUserInterface
from alexandria.results.vector_autoregression_results import *
import alexandria.console.console_utilities as cu


def vec_varma_main_code(user_inputs):
    
    
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

    # recover parameters specific to VEC/VARMA model (interface 2)
    model = ip.ext_model
    iterations = ip.ext_iterations
    burnin = ip.ext_burnin
    model_credibility = ip.ext_model_credibility
    constant = ip.ext_constant
    trend = ip.ext_trend
    quadratic_trend = ip.ext_quadratic_trend
    vec_lags = ip.vec_lags
    vec_pi1 = ip.vec_pi1
    vec_pi2 = ip.vec_pi2
    vec_pi3 = ip.vec_pi3
    vec_pi4 = ip.vec_pi4
    prior_type = ip.vec_prior_type
    error_correction_type = ip.error_correction_type
    max_cointegration_rank = ip.max_cointegration_rank
    varma_lags = ip.varma_lags
    ar_coefficients = ip.varma_ar_coefficients
    varma_pi1 = ip.varma_pi1
    varma_pi2 = ip.varma_pi2
    varma_pi3 = ip.varma_pi3
    varma_pi4 = ip.varma_pi4
    residual_lags = ip.residual_lags
    lambda1 = ip.lambda1
    lambda2 = ip.lambda2
    lambda3 = ip.lambda3    

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
    endogenous = ip.ext_endogenous
    exogenous = ip.ext_exogenous
    dates = ip.ext_dates
    forecast_dates = ip.ext_forecast_dates
    Y_p = ip.ext_Y_p
    Z_p = ip.ext_Z_p
    condition_table = ip.ext_condition_table
    shock_table = ip.ext_shock_table
    restriction_table = ip.ext_restriction_table

    # initialize timer
    ip.input_timer('start')


    #---------------------------------------------------
    # Model creation
    #--------------------------------------------------- 


    # Vector Error Correction
    if model == 1:
        from alexandria import VectorErrorCorrection
        model = VectorErrorCorrection(endogenous, exogenous = exogenous, structural_identification = structural_identification, \
                                      restriction_table = restriction_table, lags = vec_lags, max_cointegration_rank = max_cointegration_rank, \
                                      prior_type = prior_type, error_correction_type = error_correction_type, \
                                      constant = constant, trend = trend, quadratic_trend = quadratic_trend, \
                                      pi1 = vec_pi1, pi2 = vec_pi2, pi3 = vec_pi3, pi4 = vec_pi4, \
                                      credibility_level = model_credibility, iterations = iterations, burnin = burnin, verbose = progress_bar)


    # Vector Autoregressive Moving Average
    elif model == 2:
        from alexandria import VectorAutoregressiveMovingAverage
        model = VectorAutoregressiveMovingAverage(endogenous, exogenous = exogenous, structural_identification = structural_identification, \
                                                  restriction_table = restriction_table, lags = varma_lags, residual_lags = residual_lags, \
                                                  constant = constant, trend = trend, quadratic_trend = quadratic_trend, \
                                                  ar_coefficients = ar_coefficients, pi1 = varma_pi1, pi2 = varma_pi2, pi3 = varma_pi3, pi4 = varma_pi4, \
                                                  lambda1 = lambda1, lambda2 = lambda2, lambda3 = lambda3, credibility_level = model_credibility, \
                                                  iterations = iterations, burnin = burnin, verbose = progress_bar)


    #---------------------------------------------------
    # Model estimation
    #---------------------------------------------------
    

    # model estimation
    model.estimate()
    
    
    #---------------------------------------------------
    # Model application: in-sample fit and residuals
    #--------------------------------------------------- 
    
    
    # in-sample fit and residuals
    model.insample_fit()
       

    #---------------------------------------------------
    # Model application: forecasts
    #---------------------------------------------------          
        
        
    # estimate forecasts, if selected
    if forecast:
        model.forecast(forecast_periods, forecast_credibility, Z_p)
        # estimate forecast evaluation, if selected
        if forecast_evaluation:
            model.forecast_evaluation(Y_p)

        
    #---------------------------------------------------
    # Model application: impulse response function
    #--------------------------------------------------- 


    # estimate impulse response function, if selected
    if irf:
        model.impulse_response_function(irf_periods, irf_credibility)
        

    #------------------------------------------------------------
    # Model application: forecast error variance decomposition
    #------------------------------------------------------------


    # estimate forecast error variance decompositionn, if selected
    if fevd:
        model.forecast_error_variance_decomposition(irf_periods, fevd_credibility)


    #------------------------------------------------------------
    # Model application: historical decomposition
    #------------------------------------------------------------


    # estimate historical decomposition, if selected
    if hd:
        model.historical_decomposition(hd_credibility)

     
    #------------------------------------------------------------
    # Model application: conditional forecasts
    #------------------------------------------------------------


    # estimate conditional forecasts, if selected
    if conditional_forecast and conditional_forecast_type == 1:
        model.conditional_forecast(forecast_periods, conditional_forecast_credibility, condition_table, [], 1, Z_p)
    elif conditional_forecast and conditional_forecast_type == 2:  
        model.conditional_forecast(forecast_periods, conditional_forecast_credibility, condition_table, shock_table, 2, Z_p) 


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
    res = Results(model, results_information)

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
        grp = Graphics(model, graphics_information, graphics_path, True)

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
    
    
    return model

