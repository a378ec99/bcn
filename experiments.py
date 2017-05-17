import sys

import numpy as np

from utils import submit, max_measurements



def test(run):

    if run == 'figure_3a':

        seed = np.random.randint(0, 1e8)
        n_xy = 8
        
        # Figure 3a - Top, left (Measurements vs. Database Size/Dimensions)
        """
        for shape_0 in [400, 800, 1600]: # 10, 25, 30, 50, 100, 200, 

            run = 'figure_3a_dense_shape-' + str(shape_0) + '-' + str(shape_0) +  '_rank-2_shape-(' + str(shape_0) + ',' + str(shape_0) + ')_measurements-free_rep-1'
            parameters = {'run_class': 'Simulation',
                        'name': 'test_' + run + '_seed-' + str(seed),
                        'mode': 'parallel',
                        'visualization_extension': '.eps',
                        'figure_size': (8, 8),
                        'seed': seed,
                        'replicates': 1,
                        'normalize_stds': True,  # stds, correlations
                        'signal_model': ('random', 'random'),
                        'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                        'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                        'estimate': (False, False),  # pairs, stds
                        'restarts': 20,
                        'noise_amplitude': 1.0,
                        'p_random': 0.05,
                        'operator_name': 'dense',
                        'additive_noise_A_std': None,
                        'm_blocks': int(shape_0 / 2.0), # WARNING dependes on shape! Always take shape / 2.0 to get max. pairs
                        'save_signal': False,
                        'rank': 2,
                        'verbosity': 2,
                        'logverbosity': 2,
                        'maxiter': 4000,
                        'maxtime': 400,
                        'mingradnorm': 1e-16,
                        'minstepsize': 1e-16,
                        'incorrect_A_std': 1.0,
                        'correlation_strength': 1.0,
                        'additive_noise_y_std': None,
                        'unittest': False,
                        'save_run': False,
                        'save_visualize': True,
                        'known_correlations': 1.0,
                        'sparsity': 2,
                        'free_x': ('shape', [(shape_0, shape_0)]),
                        'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(0.01), np.log10(1.0), n_xy) * shape_0**2, dtype=int)))} # WARNING depends on shape! Give percentages of total (which is based on n*m or on max. pairs?)

            submit(parameters)
        
        # Figure 3a - Top, right (Measurements vs. K-sparsity)
        """
        shape_0 = 50
        """
        run = 'figure_3a_dense_shape-' + str(shape_0) + '-' + str(shape_0) +  '_rank-2_sparsity-free_measurements-free_rep-3_non-log'
        parameters = {'run_class': 'Simulation',
                      'name': 'test_' + run + '_seed-' + str(seed),
                      'mode': 'parallel',
                      'visualization_extension': '.eps',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 3,
                      'shape': (shape_0, shape_0),
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 1.0,
                      'p_random': 0.05,
                      'operator_name': 'dense',
                      'additive_noise_A_std': None,
                      'm_blocks': 5,
                      'save_signal': False,
                      'rank': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 2000,
                      'maxtime': 200,
                      'mingradnorm': 1e-14,
                      'minstepsize': 1e-14,
                      'incorrect_A_std': 1.0,
                      'correlation_strength': 1.0,
                      'additive_noise_y_std': None,
                      'unittest': False,
                      'save_run': False,
                      'save_visualize': True,
                      'known_correlations': 1.0,
                      'free_x': ('sparsity', list(np.asarray(np.linspace(1, n_xy, n_xy), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.linspace(200, 600 + 1, n_xy), dtype=int)))} # list(np.asarray(np.logspace(np.log10(200), np.log10(600), n_xy), dtype=int))

        submit(parameters)
        """
        run = 'figure_3a_dense_shape-' + str(shape_0) + '-' + str(shape_0) +  '_rank-2_sparsity-free_measurements-free_rep-3'
        parameters = {'run_class': 'Simulation',
                      'name': 'test_' + run + '_seed-' + str(seed),
                      'mode': 'parallel',
                      'visualization_extension': '.eps',
                      'figure_size': (8, 8),
                      'seed': seed,
                      'replicates': 3,
                      'shape': (shape_0, shape_0),
                      'normalize_stds': True,  # stds, correlations
                      'signal_model': ('random', 'random'),
                      'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                      'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                      'estimate': (False, False),  # pairs, stds
                      'restarts': 10,
                      'noise_amplitude': 1.0,
                      'p_random': 0.05,
                      'operator_name': 'dense',
                      'additive_noise_A_std': None,
                      'm_blocks': 5,
                      'save_signal': False,
                      'rank': 2,
                      'verbosity': 2,
                      'logverbosity': 2,
                      'maxiter': 2000,
                      'maxtime': 200,
                      'mingradnorm': 1e-14,
                      'minstepsize': 1e-14,
                      'incorrect_A_std': 1.0,
                      'correlation_strength': 1.0,
                      'additive_noise_y_std': None,
                      'unittest': False,
                      'save_run': False,
                      'save_visualize': True,
                      'known_correlations': 1.0,
                      'free_x': ('sparsity', list(np.asarray(np.linspace(1, n_xy, n_xy), dtype=int))),
                      'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(200), np.log10(600), n_xy), dtype=int)))}

        submit(parameters)
        """
        # Figure 3a - Bottom, left (Blind Recovery)
        
        run = 'figure_3a_custom_shape-' + str(shape_0) + '-' + str(shape_0) +  '_rank-free_measurements-free_rep-3_0-5'
        parameters = {'run_class': 'Simulation',
                    'name': 'test_' + run + '_seed-' + str(seed),
                    'mode': 'parallel',
                    'visualization_extension': '.eps',
                    'figure_size': (8, 8),
                    'seed': seed,
                    'replicates': 3,
                    'shape': (shape_0, shape_0),
                    'normalize_stds': True,  # stds, correlations
                    'signal_model': ('random', 'random'),
                    'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                    'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                    'estimate': (False, False),  # pairs, stds
                    'restarts': 10,
                    'noise_amplitude': 1.0,
                    'p_random': 0.05,
                    'operator_name': 'custom',
                    'additive_noise_A_std': None,
                    'm_blocks': int(shape_0 / 2.0),
                    'save_signal': False,
                    'sparsity': 2,
                    'verbosity': 2,
                    'logverbosity': 2,
                    'maxiter': 2000,
                    'maxtime': 200,
                    'mingradnorm': 1e-14,
                    'minstepsize': 1e-14,
                    'incorrect_A_std': 1.0,
                    'known_correlations': 1.0,
                    'correlation_strength': 1.0,
                    'additive_noise_y_std': None,
                    'unittest': False,
                    'save_run': False,
                    'save_visualize': True,
                    'free_x': ('rank', list(np.asarray(np.linspace(1, 20, n_xy), dtype=int))), 
                    'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(shape_0 + 0.5), np.log10(max_measurements((shape_0, shape_0))), n_xy), dtype=int)))}

        submit(parameters)

        # Figure 3a - Bottom, right (2-sparse Recovery)
        
        run = 'figure_3a_dense_shape-' + str(shape_0) + '-' + str(shape_0) +  '_rank-free_measurements-free_rep-3_0-5'
        parameters = {'run_class': 'Simulation',
                    'name': 'test_' + run + '_seed-' + str(seed),
                    'mode': 'parallel',
                    'visualization_extension': '.eps',
                    'figure_size': (8, 8),
                    'seed': seed,
                    'replicates': 3,
                    'shape': (shape_0, shape_0),
                    'normalize_stds': True,  # stds, correlations
                    'signal_model': ('random', 'random'),
                    'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                    'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                    'estimate': (False, False),  # pairs, stds
                    'restarts': 10,
                    'noise_amplitude': 1.0,
                    'p_random': 0.05,
                    'operator_name': 'dense',
                    'additive_noise_A_std': None,
                    'm_blocks': int(shape_0 / 2.0),
                    'save_signal': False,
                    'sparsity': 2,
                    'verbosity': 2,
                    'logverbosity': 2,
                    'maxiter': 2000,
                    'maxtime': 200,
                    'mingradnorm': 1e-14,
                    'minstepsize': 1e-14,
                    'incorrect_A_std': 1.0,
                    'known_correlations': 1.0,
                    'correlation_strength': 1.0,
                    'additive_noise_y_std': None,
                    'unittest': False,
                    'save_run': False,
                    'save_visualize': True,
                    'free_x': ('rank', list(np.asarray(np.linspace(1, 20, n_xy), dtype=int))), # 8
                    'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(shape_0 + 0.5), np.log10(max_measurements((shape_0, shape_0))), n_xy), dtype=int)))}

        submit(parameters)
        """
        # TODO dense with sparsity=full and free_x=size (e.g. shape_0) or % measurements






        
        
    if run == 'figure_3b':
        
        shape_0 = 50
        seed = np.random.randint(0, 1e8)
        n_xy = 8
        
        # Figure 3b - Top, left (Blind Recovery - Redundancy)
        """
        run = 'figure_3b_noise_custom_shape-' + str(shape_0) + '-' + str(shape_0) +  '_correlation_strength-free_measurements-free_rep-3'
        parameters = {'run_class': 'Simulation',
                    'name': 'final_' + run + '_seed-' + str(seed),
                    'mode': 'parallel',
                    'visualization_extension': '.eps',
                    'figure_size': (8, 8),
                    'seed': seed,
                    'replicates': 3,
                    'shape': (shape_0, shape_0),
                    'normalize_stds': True,  # stds, correlations
                    'signal_model': ('random', 'random'),
                    'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                    'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                    'estimate': (False, False),  # pairs, stds
                    'restarts': 10,
                    'noise_amplitude': 1.0,
                    'p_random': 0.05,
                    'operator_name': 'custom',
                    'm_blocks': int(shape_0 / 2.0),
                    'save_signal': False,
                    'sparsity': 2,
                    'verbosity': 2,
                    'logverbosity': 2,
                    'maxiter': 2000,
                    'maxtime': 200,
                    'mingradnorm': 1e-14,
                    'minstepsize': 1e-14,
                    'incorrect_A_std': 1.0,
                    'known_correlations': 1.0,
                    'additive_noise_y_std': None,
                    'unittest': False,
                    'save_run': False,
                    'save_visualize': True,
                    'rank': 2,
                    'additive_noise_A_std': None,
                    'free_x': ('correlation_strength', list(np.asarray(np.linspace(0.7, 1.0, n_xy)))),
                    'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(shape_0 + 0.5), np.log10(max_measurements((shape_0, shape_0))), n_xy), dtype=int)))}

        submit(parameters)
    
        # Figure 3b - Top, right (2-sparse Recovery - Additive Noise)

        run = 'figure_3b_noise_dense_shape-' + str(shape_0) + '-' + str(shape_0) +  '_additive_noise_y_std-free_measurements-free_rep-3'
        parameters = {'run_class': 'Simulation',
                    'name': 'final_' + run + '_seed-' + str(seed),
                    'mode': 'parallel',
                    'visualization_extension': '.eps',
                    'figure_size': (8, 8),
                    'seed': seed,
                    'replicates': 3,
                    'shape': (shape_0, shape_0),
                    'normalize_stds': True,  # stds, correlations
                    'signal_model': ('random', 'random'),
                    'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                    'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                    'estimate': (False, False),  # pairs, stds
                    'restarts': 10,
                    'noise_amplitude': 1.0,
                    'p_random': 0.05,
                    'operator_name': 'dense',
                    'additive_noise_A_std': None,
                    'm_blocks': int(shape_0 / 2.0),
                    'save_signal': False,
                    'sparsity': 2,
                    'verbosity': 2,
                    'logverbosity': 2,
                    'maxiter': 2000,
                    'maxtime': 200,
                    'mingradnorm': 1e-14,
                    'minstepsize': 1e-14,
                    'incorrect_A_std': 1.0,
                    'known_correlations': 1.0,
                    'correlation_strength': 1.0,
                    'unittest': False,
                    'save_run': False,
                    'save_visualize': True,
                    'rank': 2,
                    'free_x': ('additive_noise_y_std', list(np.asarray(np.linspace(0.001, 0.01, n_xy)))[::-1]),
                    'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(shape_0 + 0.5), np.log10(max_measurements((shape_0, shape_0))), n_xy), dtype=int)))}

        submit(parameters)
        """
        # Figure 3b - Bottom, left (Blind Recovery - Estimation Accuracy)

        run = 'figure_3b_noise_custom_shape-' + str(shape_0) + '-' + str(shape_0) +  '_known_correlations-free_measurements-free_rep-3_2500'
        parameters = {'run_class': 'Simulation',
                    'name': 'final_' + run + '_seed-' + str(seed),
                    'mode': 'parallel',
                    'visualization_extension': '.eps',
                    'figure_size': (8, 8),
                    'seed': seed,
                    'replicates': 3,
                    'shape': (shape_0, shape_0),
                    'normalize_stds': True,  # stds, correlations
                    'signal_model': ('random', 'random'),
                    'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                    'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                    'estimate': (False, False),  # pairs, stds
                    'restarts': 10,
                    'noise_amplitude': 1.0,
                    'p_random': 0.05,
                    'operator_name': 'custom',
                    'additive_noise_A_std': None,
                    'm_blocks': int(shape_0 / 2.0),
                    'save_signal': False,
                    'sparsity': 2,
                    'verbosity': 2,
                    'logverbosity': 2,
                    'maxiter': 2000,
                    'maxtime': 200,
                    'mingradnorm': 1e-14,
                    'minstepsize': 1e-14,
                    'incorrect_A_std': 1.0,
                    'additive_noise_y_std': None,
                    'unittest': False,
                    'save_run': False,
                    'save_visualize': True,
                    'correlation_strength': 1.0,
                    'rank': 2,
                    'free_x': ('known_correlations', list(np.asarray(np.linspace(0.8, 1.0, n_xy)))),
                    'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(shape_0 + 0.5), np.log10(2500 + 0.5), n_xy), dtype=int)))}

        submit(parameters)
        
        # Figure 3b - Bottom, right (2-sparse Recovery - Shuffled)

        run = 'figure_3b_noise_dense_shape-' + str(shape_0) + '-' + str(shape_0) +  '_known_correlations-free_measurements-free_rep-3_2500' # WARNING
        parameters = {'run_class': 'Simulation',
                    'name': 'final_' + run + '_seed-' + str(seed),
                    'mode': 'parallel',
                    'visualization_extension': '.eps',
                    'figure_size': (8, 8),
                    'seed': seed,
                    'replicates': 3,
                    'shape': (shape_0, shape_0),
                    'normalize_stds': True,  # stds, correlations
                    'signal_model': ('random', 'random'),
                    'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                    'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                    'estimate': (False, False),  # pairs, stds
                    'restarts': 10,
                    'noise_amplitude': 1.0,
                    'p_random': 0.05,
                    'operator_name': 'dense',
                    'additive_noise_A_std': None,
                    'm_blocks': int(shape_0 / 2.0),
                    'save_signal': False,
                    'sparsity': 2,
                    'verbosity': 2,
                    'logverbosity': 2,
                    'maxiter': 2000,
                    'maxtime': 200,
                    'mingradnorm': 1e-14,
                    'minstepsize': 1e-14,
                    'incorrect_A_std': 1.0,
                    'correlation_strength': 1.0,
                    'additive_noise_y_std': None,
                    'unittest': False,
                    'save_run': False,
                    'save_visualize': True,
                    'rank': 2,
                    'free_x': ('known_correlations', list(np.asarray(np.linspace(0.8, 1.0, n_xy)))),
                    'free_y': ('measurements', list(np.asarray(np.logspace(np.log10(shape_0 + 0.5), np.log10(2500 + 0.5), n_xy), dtype=int)))} # max_measurements((shape_0, shape_0)

        submit(parameters)




            
            
    if run == 'check':

        shape_0 = 50
        seed = np.random.randint(0, 1e8)
        n_xy = 8

        # Visualize with check.py
        
        run = 'check_noise_dense_shape-' + str(shape_0) + '-' + str(shape_0) +  '_known_correlations-free_measurements-free_rep-1' # WARNING
        parameters = {'run_class': 'Simulation',
                    'name': 'test_' + run + '_seed-' + str(seed),
                    'mode': 'parallel',
                    'visualization_extension': '.eps',
                    'figure_size': (8, 8),
                    'seed': seed,
                    'replicates': 1,
                    'shape': (shape_0, shape_0),
                    'normalize_stds': True,  # stds, correlations
                    'signal_model': ('random', 'random'),
                    'noise_model': 'low-rank',  # 'biclusters', image', 'euclidean'
                    'missing_model': 'no-missing',  # 'MAR', 'NMAR', 'no-missing', 'SCAN'
                    'estimate': (False, False),  # pairs, stds
                    'restarts': 10,
                    'noise_amplitude': 1.0,
                    'p_random': 0.05,
                    'operator_name': 'dense',
                    'additive_noise_A_std': None,
                    'm_blocks': int(shape_0 / 2.0),
                    'save_signal': False,
                    'sparsity': 2,
                    'verbosity': 2,
                    'logverbosity': 2,
                    'maxiter': 2000,
                    'maxtime': 200,
                    'mingradnorm': 1e-14,
                    'minstepsize': 1e-14,
                    'incorrect_A_std': 1.0,
                    'correlation_strength': 1.0,
                    'additive_noise_y_std': None,
                    'unittest': False,
                    'save_run': True, # WARNING
                    'save_visualize': True,
                    'rank': 2,
                    'free_x': ('known_correlations', [1.0]),
                    'free_y': ('measurements', [2500])}

        submit(parameters)


            
if __name__ == '__main__':
    arg = sys.argv[1]
    test(arg)





    